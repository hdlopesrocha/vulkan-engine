#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <functional>
#include <mutex>
#include "OctreeTopologyTypes.hpp"
#include "../math/geometry/Geometry.hpp"
#include "../vulkan/resources/Buffer.hpp"

class VulkanApp;

// Rolling stats for the GPU topology pass (dispatch + readback + bucketing for
// one chunk is one "call"). Logged periodically so the GPU path's cost can be
// compared against VULKAN_CPU_TOPOLOGY=1 without any UI wiring.
struct OctreeTopologyStats {
    uint64_t calls = 0;
    double avgMs = 0.0;
    double avgMsLast = 0.0;
    double avgCells = 0.0;
    double avgNodes = 0.0;
    double avgTriangles = 0.0;
};

// Destination of the GPU-resident path: the renderer's packed element pools.
// The compute passes write this chunk's deduped vertices into
// `vertexBuffer[baseVertex .. baseVertex+vertexCapacity)` in the renderer's
// 64-byte Vertex layout and its chunk-local index stream into
// `indexBuffer[firstIndex .. firstIndex+indexCapacity)`. The host never
// uploads vertex data — it only reserves the span and publishes the draw
// entry (indexCount = index stream length, vertexOffset = baseVertex).
struct OctreeTopologyTarget {
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    uint32_t baseVertex = 0;
    uint32_t vertexCapacity = 0;
    VkBuffer indexBuffer = VK_NULL_HANDLE;
    uint32_t firstIndex = 0;
    uint32_t indexCapacity = 0;
    uint32_t slotIndex = UINT32_MAX; // opaque, echoed back in the result
};

// Sizing demand handed to the target provider once the walk has run. The walk
// is submitted alone first because the index stream has exactly one entry per
// raw (pre-dedup) vertex and its span can only be reserved once that count is
// known — a worst-case reservation would have to cover the whole key space.
struct OctreeTopologyDemand {
    uint32_t rawVertexCount = 0;      // 3 per emitted triangle (== index count)
    uint32_t maxUniqueVertices = 0;   // 6 * nodeCount: the dedup key space
    uint32_t suggestedVertexCapacity = 0; // min(rawVertexCount, maxUniqueVertices)
};

// Reserves the packed span for one chunk. Called on the caller's thread
// between the walk and the dedup/index passes; returning an undersized or
// invalid target makes computeResident() fail (the caller then releases it).
using OctreeTopologyTargetProvider =
    std::function<OctreeTopologyTarget(const OctreeTopologyDemand&)>;

// Result of a resident (pool-writing) build. `bridge` is the host-visible
// mirror of exactly what landed in the pool span (renderer Vertex layout, one
// entry per deduped slot, plus the index stream): the CPU-side effect
// consumers (vegetation / lava / RT proxies) and the topology
// self-check read it.
struct OctreeTopologyResidentResult {
    Geometry bridge;              // deduped mirror of the pool span + indices
    uint32_t poolVertexCount = 0; // deduped vertices written into the pool span
    uint32_t indexCount = 0;      // index stream length (== raw vertex count)
    uint32_t slotIndex = UINT32_MAX;
    bool ok = false;              // pool span fully written: safe to draw
};

// GPU triangle-topology service: dispatches
// shaders/computation/octree/OctreeTriangles.comp (the exact port of
// Octree::iterateTriangles) over a flattened octree neighborhood. Two output
// modes share one pipeline:
//   compute()         — arena readback only (CPU consumers that still need
//                       the walk's raw triangle stream on the host);
//   computeResident() — dedup into the renderer's packed pools, no vertex data
//                       uploaded (the app's chunk path).
class OctreeTopologyCompute {
public:
    static OctreeTopologyCompute* globalInstance __attribute__((weak));
    static OctreeTopologyCompute* getGlobal() __attribute__((weak)) { return globalInstance; }

    OctreeTopologyCompute();
    void init(VulkanApp* app);
    void destroy(VulkanApp* app);

    // Synchronous, worker-thread safe (serialized). Returns the walk's raw
    // triangle stream; when the fixed output arena overflows, `truncated` is
    // set and the caller must fall back to the CPU walker.
    OctreeTopologyResult compute(const OctreeTopologyInput& input) __attribute__((weak));

    // Synchronous, worker-thread safe (serialized). Runs the walk, asks
    // `provider` for a span sized from the walk's actual raw vertex count and
    // then dedups straight into the renderer's packed pools (no vertex data is
    // uploaded). `result.ok` is false when the walk could not be represented
    // (key capacity or span overflow): the caller must drop the chunk — there
    // is no CPU fallback on this path.
    OctreeTopologyResidentResult computeResident(
        const OctreeTopologyInput& input,
        const OctreeTopologyTargetProvider& provider) __attribute__((weak));

    // Dedup keys per node: one per (node, triplanar plane) pair. Bounds both
    // the dedup table and the number of distinct vertices a chunk can produce.
    static constexpr uint32_t kVertexKeysPerNode = 6;

    OctreeTopologyStats stats() __attribute__((weak));

    VulkanApp* app = nullptr;
    std::mutex mutex;

private:
    // Grows the resident mirror (bridgeVerts/bridgeIdx) to hold at least
    // `uniqueCapacity` vertices and `indexCapacity` indices.
    void ensureBridgeCapacity(size_t uniqueCapacity, size_t indexCapacity);
    void ensureCapacity(size_t nodeCount, size_t cellCount, size_t vertexCount,
                        size_t keyCount);
    // Uploads the flattened neighborhood and resets the counters. Caller must
    // hold `mutex` and have sized the buffers for this input.
    void uploadInput(const OctreeTopologyInput& input);
    // Points the descriptor set at the arena (nullptr) or at the resident
    // target's pools.
    void writeDescriptors(const OctreeTopologyTarget* target);

    uint64_t statCalls = 0;
    double statMs = 0.0;
    double statMsLast = 0.0;
    double statCells = 0.0;
    double statNodes = 0.0;
    double statTriangles = 0.0;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSet descSet = VK_NULL_HANDLE;

    size_t nodeCap = 0;
    size_t cellCap = 0;
    size_t vertexCap = 0;
    size_t keyCap = 0;
    size_t tableCap = 0;
    size_t bridgeVertCap = 0;
    size_t bridgeIdxCap = 0;
    Buffer bSdf0{}, bSdf1{}, bCube{}, bMeta{}, bHsv{}, bChild03{}, bChild47{};
    Buffer bCells{}, bPos{}, bNrm{}, bOutHsv{}, bCounter{};
    Buffer bRawKeys{}, bDedupTable{}, bBridgeVerts{}, bBridgeIdx{};
};

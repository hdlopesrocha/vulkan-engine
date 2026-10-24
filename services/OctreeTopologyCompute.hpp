#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <functional>
#include <mutex>
#include "OctreeTopologyTypes.hpp"
#include "OctreeTopologyStats.hpp"
#include "OctreeTopologyTarget.hpp"
#include "OctreeTopologyDemand.hpp"
#include "OctreeTopologyResidentResult.hpp"
#include "../vulkan/resources/Buffer.hpp"

class VulkanApp;

// Reserves the packed span for one chunk. Called on the caller's thread
// between the walk and the dedup/index passes; returning an undersized or
// invalid target makes computeResident() fail (the caller then releases it).
using OctreeTopologyTargetProvider =
    std::function<OctreeTopologyTarget(const OctreeTopologyDemand&)>;

// GPU triangle-topology service: dispatches
// shaders/computation/octree/OctreeTriangles.comp (the exact port of the
// adaptive surface-nets walker) over a flattened octree neighborhood and
// dedups/writes straight into the renderer's packed pools. There is no arena,
// no host-side vertex stream and no CPU fallback.
class OctreeTopologyCompute {
public:
    static OctreeTopologyCompute* globalInstance __attribute__((weak));
    static OctreeTopologyCompute* getGlobal() __attribute__((weak)) { return globalInstance; }

    OctreeTopologyCompute();
    void init(VulkanApp* app);
    void destroy(VulkanApp* app);

    // Synchronous, worker-thread safe (serialized). Runs the walk, asks
    // `provider` for a span sized from the walk's actual raw vertex count and
    // then dedups straight into the renderer's packed pools. `result.ok` is
    // false when the walk could not be represented (key capacity or span
    // overflow): the caller drops the chunk — there is no CPU fallback.
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
    void ensureCapacity(size_t nodeCount, size_t cellCount, size_t keyCount);
    // Uploads the flattened neighborhood and resets the counters. Caller must
    // hold `mutex` and have sized the buffers for this input.
    void uploadInput(const OctreeTopologyInput& input);
    // Points the descriptor set at the resident target's pools (nullptr for
    // the walk pass, which never touches the pool bindings).
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
    size_t keyCap = 0;
    size_t tableCap = 0;
    Buffer bSdf0{}, bSdf1{}, bCube{}, bMeta{}, bHsv{}, bChild03{}, bChild47{};
    Buffer bCells{}, bCounter{};
    Buffer bRawKeys{}, bDedupTable{};
};

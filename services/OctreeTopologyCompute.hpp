#pragma once
#include <vulkan/vulkan.h>
#include <cstdint>
#include <functional>
#include <mutex>
#include "OctreeTopologyTypes.hpp"
#include "OctreeTopologyTarget.hpp"
#include "OctreeTopologyDemand.hpp"
#include "OctreeTopologyResidentResult.hpp"
#include "../vulkan/resources/Buffer.hpp"

class VulkanApp;

// Reserves the packed span for one chunk from the CPU cross-edge capacity
// bounds. Runs on the caller's thread before the geometry dispatch; returning
// an undersized or invalid target makes computeResident() fail (the caller then
// releases it).
using OctreeTopologyTargetProvider =
    std::function<OctreeTopologyTarget(const OctreeTopologyDemand&)>;

// Releases a reservation that will not be used. Called between overflow-growth
// retries so only the affected chunk's span is re-reserved; the callback is
// supplied by the owner of the target provider (the renderer).
using OctreeTopologyReleaseCallback = std::function<void(uint32_t slotIndex)>;

// GPU triangle-topology service: dispatches
// shaders/computation/octree/OctreeTriangles.comp (the exact port of the
// adaptive surface-nets walker) over a flattened octree neighborhood and
// writes straight into the renderer's packed pools. There is no arena,
// no host-side vertex stream and no CPU fallback. One dispatch per chunk
// normally: the CPU cross-edge count sizes both spans, the shader appends
// indices directly and writes each referenced vertex once.
class OctreeTopologyCompute {
public:
    static OctreeTopologyCompute* globalInstance __attribute__((weak));
    static OctreeTopologyCompute* getGlobal() __attribute__((weak)) { return globalInstance; }

    OctreeTopologyCompute();
    ~OctreeTopologyCompute();
    void init(VulkanApp* app);

    // Synchronous, worker-thread safe (serialized). Asks `provider` for a span
    // sized from the CPU cross-edge bounds, then dispatches the geometry pass.
    // On an index/vertex span overflow the shader still reports the exact
    // required counts (without OOB writes); the service releases the span and
    // asks the provider for the grown capacity, retrying only this chunk.
    // `result.ok` is false when the chunk cannot be represented (fixed-array
    // overflow, invalid slots, or a provider that cannot fit the required
    // span): the caller drops the chunk — there is no CPU fallback.
    OctreeTopologyResidentResult computeResident(
        const OctreeTopologyInput& input,
        const OctreeTopologyTargetProvider& provider,
        const OctreeTopologyReleaseCallback& release = {}) __attribute__((weak));

    VulkanApp* app = nullptr;
    std::mutex mutex;

private:
    void ensureCapacity(size_t nodeCount, size_t cellCount);
    // Uploads the flattened neighborhood and resets the counters. Caller must
    // hold `mutex` and have sized the buffers for this input.
    void uploadInput(const OctreeTopologyInput& input);
    // Points every descriptor binding at the resident target's pools and the
    // per-dispatch node-written flags.
    void writeDescriptors(const OctreeTopologyTarget& target);

    // One chunk's VULKAN_SETUP_DIAG sample. Plain data; computeResident() holds
    // `mutex` for its whole body, so no field needs its own synchronization.
    struct ProfileSample {
        double totalMs = 0.0;      // computeResident, including mutex wait
        double uploadMs = 0.0;
        double descriptorMs = 0.0; // all descriptor writes
        double providerMs = 0.0;   // span reservation (and retries)
        double dispatchMs = 0.0;   // geometry dispatch + fence
        uint64_t nodes = 0;
        uint64_t cells = 0;
        uint64_t chunks = 0;
        uint64_t indices = 0;
        uint64_t vertices = 0;     // distinct vertex payloads written
        uint64_t crossEdges = 0;
        uint64_t vertexCapacity = 0;
        uint64_t indexCapacity = 0;
        uint64_t uploadedBytes = 0;
        uint64_t descriptorWrites = 0;
        uint64_t retries = 0;
        uint64_t indexOverflow = 0;
        uint64_t vertexOverflow = 0;
        uint64_t findCellCalls = 0;  // findCellAt descents (diag only)
        uint64_t reuseHits = 0;      // quadrant lookups avoided by AABB reuse (diag only)
    };
    void recordProfile(const ProfileSample& s);
    void printProfileSummary();
    bool profileSummaryPrinted = false;

    // Cached VULKAN_SETUP_DIAG lookup: zero clock reads and no allocation when
    // the profiler is off.
    bool setupDiag = false;
    ProfileSample profTotal{};
    ProfileSample profWindow{};
    uint64_t statDescriptorWrites = 0; // actual bindings written (M6)

    // Bumped by ensureCapacity() whenever an input buffer is recreated; the
    // descriptor path rewrites bindings 0-9 only when it changes (M6).
    // lastWrittenGeneration starts "impossible" so the first call writes all 12.
    uint64_t bufferGeneration = 1;
    uint64_t lastWrittenGeneration = ~0ull;

    uint64_t statCalls = 0;
    double statMs = 0.0;
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
    Buffer bSdf0{}, bSdf1{}, bCube{}, bMeta{}, bHsv{}, bChild03{}, bChild47{};
    Buffer bCells{}, bCounter{};
    Buffer bNodeWritten{};
};

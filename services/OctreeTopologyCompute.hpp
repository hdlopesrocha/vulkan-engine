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
    ~OctreeTopologyCompute();
    void init(VulkanApp* app);

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

    // One chunk's VULKAN_SETUP_DIAG sample. Plain data; computeResident() holds
    // `mutex` for its whole body, so no field needs its own synchronization.
    struct ProfileSample {
        double totalMs = 0.0;      // computeResident, including mutex wait
        double uploadMs = 0.0;
        double descriptorMs = 0.0; // both writeDescriptors calls
        double walkMs = 0.0;       // walk submission + fence
        double providerMs = 0.0;
        double phase2Ms = 0.0;     // fill/dedup/index dispatches + barriers + fence
        uint64_t nodes = 0;
        uint64_t cells = 0;
        uint64_t chunks = 0;
        uint64_t rawVertices = 0;
        uint64_t uniqueVertices = 0;
        uint64_t uploadedBytes = 0;
        uint64_t descriptorWrites = 0;
        uint64_t retries = 0;
        uint64_t keyCapacity = 0;
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
    // descriptor path rewrites bindings 0-10 only when it changes (M6).
    // lastWrittenGeneration starts "impossible" so the first call writes all 13.
    uint64_t bufferGeneration = 1;
    uint64_t lastWrittenGeneration = ~0ull;

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

#pragma once
#include <vulkan/vulkan.h>
#include <atomic>
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

// GPU triangle-topology service: optionally dispatches
// shaders/computation/octree/OctreeVirtualRefine.comp to refine the QEF
// position of each surface node, then dispatches
// shaders/computation/octree/OctreeTriangles.comp (the exact port of the
// adaptive surface-nets walker) over a flattened octree neighborhood and
// writes straight into the renderer's packed pools. There is no arena,
// no host-side vertex stream and no CPU fallback. One geometry dispatch per
// chunk normally: the CPU cross-edge count sizes both spans, the shader
// appends indices directly and writes each referenced vertex once.
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

    // Runtime switch for the GPU virtual-refinement pass.  The constructor
    // seeds it from VULKAN_VIRTUAL_REFINE (default on); setting it to false
    // restores the exact pre-refinement Dual-Contouring path on the next chunk
    // (no reallocation, the geometry pass simply ignores refinedPos[]).
    void setVirtualRefinementEnabled(bool enabled) { refineEnabled.store(enabled); }
    bool virtualRefinementEnabled() const { return refineEnabled.load(); }

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
        // Virtual adaptive refinement (diag only).
        double refineGpuMs = 0.0;    // timestamp query around the refine dispatch
        uint64_t refineNodes = 0;    // slot-bearing surface nodes processed
        uint64_t refineRefined = 0;  // nodes that ran at least one subdivision level
        uint64_t refineCells = 0;    // virtual child cells generated
        uint64_t refineEvals = 0;    // trilinear corner evaluations
        uint64_t refineIntersections = 0;
        uint64_t refineSelected = 0; // cells selected for a deeper level
        uint64_t refineMaxDepth = 0;
        uint64_t refineSkippedAccurate = 0;
        uint64_t refineSkippedUnsupported = 0;
        uint64_t refineOverflow = 0;
        uint64_t refineBaseResidual = 0; // sum of fixed-point (1e3) normalized values
        uint64_t refineRefResidual = 0;
        uint64_t refineBaseDist = 0;
        uint64_t refineRefDist = 0;
        uint64_t refineAbsSdf = 0;
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

    void initRefinePipeline(VkDevice device, const char* modulePath);

    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    // Virtual-refinement pipeline/push-constant layout.  It shares descLayout
    // and descSet with the geometry pass (same bindings, different PC block).
    VkPipeline refinePipeline = VK_NULL_HANDLE;
    VkPipelineLayout refineLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSet descSet = VK_NULL_HANDLE;

    // Settings (VULKAN_VIRTUAL_REFINE / VULKAN_REFINE_*).  All read under
    // `mutex` in computeResident; the enabled flag is atomic so a widget can
    // toggle it without waiting for a chunk build.
    std::atomic<bool> refineEnabled{true};
    uint32_t refineMaxDepth = 2;
    uint32_t refineBudget = 8;
    float refineMinPriority = 0.35f;
    float refineFeatureSlope = 0.90f;
    float refineTolerance = 0.010f;
    bool refineProject = false;
    bool refineTimestamps = false; // device supports diagnostic timestamps
    float timestampPeriodNs = 0.0f;

    size_t nodeCap = 0;
    size_t cellCap = 0;
    Buffer bSdf0{}, bSdf1{}, bCube{}, bMeta{}, bHsv{}, bChild03{}, bChild47{};
    Buffer bCells{}, bCounter{};
    Buffer bNodeWritten{};
    // vec4 per node (xyz refined position, w valid flag), device-local.
    Buffer bRefinedPos{};
    // 16 host-visible diagnostic uints; written by OctreeVirtualRefine.comp.
    Buffer bRefineStats{};
};

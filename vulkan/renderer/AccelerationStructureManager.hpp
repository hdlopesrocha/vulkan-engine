#pragma once
// AccelerationStructureManager — incremental BLAS/TLAS management built
// directly on the engine's GPU-resident packed geometry pools.
//
// Design (no full rebuilds, no CPU readbacks):
//  - One BLAS per active IndirectRenderer slot (chunk). The BLAS geometry
//    points INTO the shared solid/water vertex+index buffers via device
//    addresses + (baseVertex, firstIndex) sub-ranges, so no geometry is copied.
//  - One TLAS with one instance per BLAS (identity transform; scene geometry
//    is world-space). instanceCustomIndexEXT routes instance -> layer/slot ->
//    material (bit 31 = MATERIAL_WATER, bit 30 = vegetation/alpha-test).
//  - Dirty tracking: syncFromScene() snapshots the active slot geometries and
//    diffs them against the previous frame. Only added/modified slots rebuild
//    their BLAS; the TLAS rebuilds only when the instance set or a rebuilt
//    BLAS address changes. Steady state issues zero AS builds.
//  - All builds are recorded into the caller's command buffer with
//    Synchronization2 barriers (AS_BUILD -> RAY_TRACING_SHADER) and use a
//    shared scratch buffer + deferred destruction via deferDestroyUntilFence.
//    No vkDeviceWaitIdle/vkQueueWaitIdle appears in the frame loop.

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

#include "../Buffer.hpp"
#include "RayTracingSupport.hpp"

class VulkanApp;
class IndirectRenderer;

class AccelerationStructureManager {
public:
    // Frame-slot count for host-written inputs (instance + slot-meta buffers).
    // Matches VulkanApp::MAX_FRAMES_IN_FLIGHT: each slot is reused only after
    // its frame fence signals, so a slot's host rewrite can never race an
    // in-flight dispatch reading it. Must stay in lockstep with the caller's
    // slot index (frameIndex % kFrameSlots).
    static constexpr uint32_t kFrameSlots = 3;

    AccelerationStructureManager() = default;
    ~AccelerationStructureManager() = default;
    AccelerationStructureManager(const AccelerationStructureManager&) = delete;
    AccelerationStructureManager& operator=(const AccelerationStructureManager&) = delete;

    // Returns false when the device has no RT support (caller keeps rasterizer).
    bool init(VulkanApp* app);
    void cleanup(VulkanApp* app);
    bool isReady() const { return ready_; }

    // Per-layer slot snapshot consumed by syncFromScene.
    struct LayerSnapshot {
        const IndirectRenderer* ir = nullptr;
        bool isWater = false;
        bool isVegetation = false;
    };

    // LoD rung selection inputs (mirror indirect.comp's push constants).
    // The TLAS instances exactly the rung the GPU cull would draw per region,
    // so coarse ancestor rungs never smear over fine detail in RT output.
    struct LodSelect {
        glm::vec3 camPos = glm::vec3(0.0f);
        float lodBias = 8.0f;
        uint32_t maxTargetLod = 16;
    };

    // Clipmap band gate, bit-faithful mirror of indirect.comp: keep a chunk
    // iff its 0-based rung matches floor(dist(cam, base+cell/2)/(baseCell*bias))
    // clamped to [0, min(maxLevel, maxTargetLod)]. cellSize<=0 falls through
    // (kept), exactly like the shader's legacy path.
    static bool lodRungSelected(int entryLevel, float cellSize, int maxLevel,
                                const glm::vec3& base, const glm::vec3& camPos,
                                float lodBias, uint32_t maxTargetLod) {
        if (!(cellSize > 0.0f)) return true;
        if (!(lodBias > 0.0f)) return true;
        const float baseCell = cellSize / exp2f(static_cast<float>(entryLevel > 0 ? entryLevel : 0));
        const glm::vec3 center = base + 0.5f * cellSize;
        const float band = glm::distance(camPos, center) / (baseCell * lodBias);
        float selected = floorf(band);
        if (selected > static_cast<float>(maxLevel)) selected = static_cast<float>(maxLevel);
        if (selected < 0.0f) selected = 0.0f;
        if (selected > static_cast<float>(maxTargetLod)) selected = static_cast<float>(maxTargetLod);
        return static_cast<float>(entryLevel) == selected;
    }

    // Diff the live slot geometries against the previous snapshot. Returns true
    // when any BLAS or the TLAS needs a (re)build. Must be called on the main
    // thread once per frame before buildPending(). Never touches the GPU.
    bool syncFromScene(VulkanApp* app,
                       const IndirectRenderer* solidIR,
                       const IndirectRenderer* waterIR,
                       const IndirectRenderer* vegHintIR,
                       const LodSelect& lod,
                       uint32_t frameSlot);

    // Record pending BLAS builds + (if needed) the TLAS build into cmd.
    // Emits Synchronization2 barriers so the TLAS + ray dispatch that follow
    // in the same command buffer observe completed builds. No-ops when nothing
    // is dirty. Returns the number of BLAS builds recorded (+1 if TLAS built).
    uint32_t buildPending(VulkanApp* app, VkCommandBuffer cmd, uint32_t frameSlot);

    // Newest built TLAS across all slots (UINT32_MAX slot = none yet). Every
    // dispatch uses this object, so a selection change is visible in ALL
    // frames immediately — not just the rebuilding slot's frames (which would
    // flicker between old and new rungs indefinitely).
    VkAccelerationStructureKHR tlasFresh() const {
        return freshTlasSlot_ < kFrameSlots ? tlasAS_[freshTlasSlot_] : VK_NULL_HANDLE;
    }
    uint32_t tlasInstanceCount() const { return tlasInstanceCount_; }

    struct SlotMeta {
        uint32_t baseVertex = 0;
        uint32_t firstIndex = 0;
        uint32_t vertexCount = 0;
        uint32_t indexCount = 0;
    };
    // Packed per-layer slot meta for the closest-hit vertex fetch
    // (indexed by TLAS instanceCustomIndex slot id). Separate arrays per layer
    // keep the shader's water/solid buffer selection trivial. Per frame slot:
    // the caller's descriptor set for slot s must bind metaBuffers_[s].
    const Buffer& solidMetaBuffer(uint32_t slot) const { return solidMetaBufs_[slot % kFrameSlots]; }
    const Buffer& waterMetaBuffer(uint32_t slot) const { return waterMetaBufs_[slot % kFrameSlots]; }
    const Buffer& instanceBuffer(uint32_t slot) const { return instanceBufs_[slot % kFrameSlots]; }
    uint32_t solidSlotCapacity() const { return solidSlotCap_; }
    uint32_t waterSlotCapacity() const { return waterSlotCap_; }

    struct Stats {
        uint32_t blasCount = 0;
        uint32_t tlasInstances = 0;
        // Columns (shared base anchor + layer) with >1 selected rung. Nonzero
        // means rung overlap survived selection — the direct measure of the
        // "all levels rendered" defect. Zero with bad visuals points elsewhere.
        uint32_t overlapColumns = 0;
        uint32_t pendingBlasBuilds = 0;
        bool tlasDirty = false;
        float lastBlasBuildMs = 0.0f;
        float lastTlasBuildMs = 0.0f;
        uint64_t totalBlasBuilds = 0;
        uint64_t totalTlasBuilds = 0;
    };
    Stats stats() const;

private:
    struct BlasKey {
        uint8_t layer; // 0 = solid, 1 = water
        uint32_t slot;
        bool operator==(const BlasKey& o) const { return layer == o.layer && slot == o.slot; }
    };
    struct BlasKeyHash {
        size_t operator()(const BlasKey& k) const noexcept {
            return (static_cast<size_t>(k.layer) << 32) ^ k.slot;
        }
    };
    struct BlasEntry {
        VkAccelerationStructureKHR as = VK_NULL_HANDLE;
        Buffer storage{};
        VkDeviceAddress address = 0;
        VkDeviceSize storageSizeForBuild_ = 0;
        uint32_t baseVertex = 0;
        uint32_t vertexCount = 0;
        uint32_t firstIndex = 0;
        uint32_t indexCount = 0;
        // LoD band inputs (refreshed every sync; drive TLAS rung selection).
        int lodLevel = 0;
        float cellSize = 0.0f;
        glm::vec3 lodBase = glm::vec3(0.0f);
        int maxLevel = 16;
        bool dirty = true;
    };

    bool ensureScratch(VulkanApp* app, VkDeviceSize bytes, uint32_t frameSlot);
    bool ensureInstanceBuffer(VulkanApp* app, uint32_t instances);
    bool ensureMetaBuffer(VulkanApp* app, std::array<Buffer, kFrameSlots>& bufs, uint32_t& cap, uint32_t slots);
    void destroyBlas(VulkanApp* app, BlasEntry& e);
    // Rebuild the TLAS source list from BLAS entries passing the LoD gate.
    // Returns an FNV-1a hash of the selected (layer, slot) set so callers can
    // detect selection changes (camera motion) without comparing vectors.
    uint64_t rebuildTlasInstances();
    // Defer a replaced buffer's destruction until no tracked command buffer
    // is pending (in-flight frames may still read it). Never destroy inline.
    void retireBuffer(VulkanApp* app, Buffer b);

    VulkanApp* app_ = nullptr;
    bool ready_ = false;

    std::unordered_map<BlasKey, BlasEntry, BlasKeyHash> blases_;
    // Pending build list (keys into blases_).
    std::vector<BlasKey> pendingBlas_;

    struct TlasInstanceSrc {
        BlasKey key;
        VkDeviceAddress blasAddress = 0;
        bool isWater = false;
        bool isVegetation = false;
    };
    std::vector<TlasInstanceSrc> tlasSrc_;
    bool tlasDirty_ = true;

    // Per-slot TLAS object+storage: rebuilt in place on selection change, so
    // the same slot is reused only after its frame fence signals — the same
    // fence-ordered reuse as the engine's per-slot cull buffers. Bounded (3x),
    // unlike per-rebuild allocation during camera motion.
    std::array<VkAccelerationStructureKHR, kFrameSlots> tlasAS_{};
    std::array<Buffer, kFrameSlots> tlasStorage_{};
    std::array<VkDeviceSize, kFrameSlots> tlasStorageSize_{};
    // Slot holding the freshest TLAS build (UINT32_MAX = none yet). Builds
    // land in the current frame's slot (fence-ordered reuse); dispatches read
    // through tlasFresh(), so stale slots can never surface.
    uint32_t freshTlasSlot_ = UINT32_MAX;
    // Per-slot TLAS instance inputs (host-rewritten every TLAS build; a
    // single shared buffer would race in-flight dispatches reading it).
    std::array<Buffer, kFrameSlots> instanceBufs_{};
    uint32_t tlasInstanceCount_ = 0;
    uint32_t instanceCap_ = 0;

    // Per-slot build scratch (same fence-ordered reuse rationale as the TLAS
    // above; a single shared scratch would need cross-submit chaining).
    // Grows rarely per slot; replaced buffers retire deferred.
    std::array<Buffer, kFrameSlots> scratchs_{};
    std::array<VkDeviceSize, kFrameSlots> scratchSizes_{}; // allocated bytes (incl. alignment slack)
    // Aligned build addresses within each scratch (>= minAccelerationStructure-
    // ScratchOffsetAlignment, a hard VUID on scratchData.deviceAddress).
    std::array<VkDeviceAddress, kFrameSlots> scratchAddrs_{};
    std::array<VkDeviceSize, kFrameSlots> scratchUsables_{}; // usable bytes from each addr
    VkDeviceSize scratchAlign_ = 256; // queried in init(), spec-safe default

    // Per-slot vertex-fetch metadata (same host-race rationale as above).
    std::array<Buffer, kFrameSlots> solidMetaBufs_{};
    std::array<Buffer, kFrameSlots> waterMetaBufs_{};
    uint32_t solidSlotCap_ = 0;
    uint32_t waterSlotCap_ = 0;

    // NOTE: no cross-submit chaining primitive lives here anymore. Earlier
    // revisions used a persistent VkEvent (Wait/Set/Reset); the layer's event
    // state tracking proved too brittle across frames, so all mutable AS
    // memory is now either per-slot (fence-ordered reuse) or freshly
    // allocated per rebuild with deferred retirement. Intra-CB barriers still
    // order builds/dispatch within one command buffer.
    // Shared-pool device addresses captured at sync time (stable handles, so
    // the build step can point BLAS geometries into the pools without copies).
    VkDeviceAddress solidVtxAddrForBuild_ = 0;
    VkDeviceAddress solidIdxAddrForBuild_ = 0;
    VkDeviceAddress waterVtxAddrForBuild_ = 0;
    VkDeviceAddress waterIdxAddrForBuild_ = 0;
    // Last LoD selection (camera/bias params + selected-set hash). A hash
    // change means the camera moved across a rung boundary and the TLAS must
    // rebuild even with static geometry.
    LodSelect curLod_{};
    uint64_t lastSelectionHash_ = 0;
    uint32_t overlapColumns_ = 0;

    // CPU-side timing accumulators (host-clocked around sync/build calls).
    float lastBlasBuildMs_ = 0.0f;
    float lastTlasBuildMs_ = 0.0f;
    uint64_t totalBlasBuilds_ = 0;
    uint64_t totalTlasBuilds_ = 0;
};

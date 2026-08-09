// IndirectRenderer.hpp
#pragma once

#include "Renderer.hpp"

#include "../VulkanApp.hpp"
#include "../TrackedHandle.hpp"
#include "../../math/Geometry.hpp"
#include <glm/glm.hpp>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <cstdint>

#include <array>
#include <functional>
#include "CommandBufferState.hpp"
#include "../streaming/StreamCommon.hpp"
#include "../includes/locations.hpp"
#include "SlotAllocator.hpp"
#include "PackedSpaceAllocator.hpp"

namespace streaming { class UploadManager; }

// Manages a single large vertex/index/indirect buffer and provides a simple
// CPU-side allocator for adding/removing meshes. Draws are performed via
// vkCmdDrawIndexedIndirect (one indirect command per mesh). The allocator is
// append-first and supports reclamation on remove (simple free list rebuild).
//
// STABLE SLOT MODE (preferred):
// Call initSlots() once with the maximum expected active chunk count and the
// TOTAL vertex/index element budgets, then use addMeshSlotted()/
// removeMeshSlotted(). Each chunk gets a fixed draw-entry block (a "slot")
// that is never compacted, while its vertex/index data is packed into the
// shared element pools by PackedSpaceAllocator — multiple chunks share the
// pool space ("packed slots"), so the active chunk count is bounded by total
// geometry bytes, not by a fixed slot count. Updating a chunk only touches
// its own block and level ranges — there is NO global rebuild. The
// indirect/bounds buffer layout is stable, so GPU culling continues to work
// without interruption.
class IndirectRenderer : public Renderer {
public:
    static constexpr uint32_t MAX_CULL_FRAMES = 3;
        // Allow external code to force the dirty flag
        void setDirty(bool value) { dirty = value; }
    // Upload vertex and index data for a single mesh (coalesced into one transfer)
    bool uploadMesh(VulkanApp* app, uint32_t meshId);
    // Upload vertex and index data for a batch of meshes in a single staging
    // buffer / command buffer submission.  Coalescing amortizes submission
    // overhead and removes the per-chunk fence stall that serialized chunk
    // uploads when processed one-by-one (see perf_report).
    bool uploadMeshes(VulkanApp* app, const std::vector<uint32_t>& meshIds, float priority = 0.0f);

    // Route incremental per-mesh GPU copies through the shared async
    // UploadManager (the real transfer engine) instead of the single-slot
    // pendingTransfer path. When set, uploadMeshes()/uploadMesh() enqueue an
    // UploadJob (no per-frame cap, K concurrent staging slots) and publish each
    // mesh's indirect/bounds meta entry when its own transfer retires. Passing
    // nullptr restores the legacy staging-ring path.
    void setUploadManager(streaming::UploadManager* mgr, streaming::StreamCategory category) {
        uploadMgr_ = mgr;
        streamCategory_ = category;
    }
    bool hasUploadManager() const { return uploadMgr_ != nullptr; }
    // Write all mesh indirect/model/bounds buffers for all active meshes
    void uploadMeshMetaBuffers(VulkanApp* app);
    struct MeshInfo {
        uint32_t id = UINT32_MAX;
        uint32_t baseVertex = 0;
        uint32_t vertexCount = 0;
        uint32_t firstIndex = 0;
        uint32_t indexCount = 0;
        glm::vec4 boundsMin = glm::vec4(0.0f); // object-space AABB min (xyz)
        glm::vec4 boundsMax = glm::vec4(0.0f); // object-space AABB max (xyz)
        VkDeviceSize indirectOffset = 0; // byte offset into indirect buffer
        uint32_t drawIndex = UINT32_MAX; // position in indirectCommands list
        uint32_t slotIndex = UINT32_MAX; // stable slot (if using slotted mode)
        bool active = false;
        // Single chunk mesh (slotted mode): one packed span in the shared
        // vertex/index element pools, one draw entry per chunk. Chunks arrive
        // one by one — each chunk publishes exactly one mesh.
        struct LevelData {
            bool allocated = false;
            uint32_t baseVertex = 0;   // absolute element offset (mergedVertices)
            uint32_t vertexCount = 0;
            uint32_t firstIndex = 0;   // absolute element offset (mergedIndices)
            uint32_t indexCount = 0;
            int level = 0;             // 0-based LoD level (0 = frontier chunk)
            // Ranges of the mesh's PREVIOUS geometry, freed once the
            // replacement upload completes (deferred — in-flight frames may
            // still reference them). UINT32_MAX base = nothing pending.
            uint32_t oldVertexBase = UINT32_MAX;
            uint32_t oldVertexCount = 0;
            uint32_t oldIndexBase = UINT32_MAX;
            uint32_t oldIndexCount = 0;
            glm::vec4 boundsMin = glm::vec4(0.0f);
            glm::vec4 boundsMax = glm::vec4(0.0f);
        };
        LevelData level_ = {};
        // NOTE: per-mesh buffers removed — meshes are packed into the merged buffers
    };

    IndirectRenderer();
    ~IndirectRenderer();

    void init();
    void cleanup(VulkanApp* app) override;

    // ── Legacy append-based API (triggers full rebuild) ──
    // Add mesh and return mesh id.
    uint32_t addMesh(const Geometry& mesh);
    // Add mesh with a custom ID (e.g., node ID from octree). If mesh with this ID exists, it is replaced.
    uint32_t updateMesh(const Geometry& mesh, uint32_t customId);
    void removeMesh(uint32_t meshId);
    // Remove all meshes and reset GPU write tracking.
    void removeAllMeshes();
    // Rebuild GPU backing buffers from current CPU mesh list.
    void rebuild(VulkanApp* app);

    // ── Stable slot-based API (no global rebuilds) ──
    // Pre-allocate the packed element pools and the draw-entry pool and
    // create GPU buffers sized to capacity. `maxActiveChunks` is the maximum
    // number of concurrently active chunks (each owns one draw entry);
    // `totalVertexBytes`/`totalIndexBytes` are the TOTAL shared element
    // budgets (vertex + index pools) — chunks pack into them, so the element
    // budgets are hard caps just like the slot count. Must be called once on
    // the main thread with no pending GPU work.
    void initSlots(VulkanApp* app,
                   uint32_t maxActiveChunks,
                   uint32_t totalVertexBytes,
                   uint32_t totalIndexBytes);

    // Add or update a mesh in a stable slot. The chunk's draw entry is
    // allocated on first use and freed on removal; the vertex/index data is
    // packed into the shared element pools. Returns the stable slot index,
    // or UINT32_MAX on failure (pool exhausted or element budget exceeded).
    // Each chunk owns exactly one draw entry (slot index == entry index).
    // Re-publishing an already-allocated chunk allocates a NEW span and
    // defers the free of the old span until the replacement upload completes
    // (in-flight frames may still reference the old data).
    // `cubeMin`/`cubeMax`: the chunk's OWN cube bounds. When provided they
    // are published as the draw entry's bounds triple instead of the mesh
    // AABB, keeping frustum culling conservative-correct for edge-surface
    // chunks.
    // `level`: the chunk's 0-based LoD level (0 = frontier chunks, the
    // finest; higher = coarser ancestor cells). Stored in the entry's bounds
    // meta and used by the GPU cull to keep only the chunk level matching
    // the camera distance band.
    uint32_t addMeshSlotted(const Geometry& mesh, uint32_t chunkId,
                            const glm::vec3* cubeMin = nullptr, const glm::vec3* cubeMax = nullptr,
                            int level = 0);
    void removeMeshSlotted(uint32_t slotIndex);

    // Upload a single mesh's vertex/index data to the GPU, and write its
    // indirect command + bounds into the host-visible metadata buffers.
    // This is the per-chunk equivalent of a full rebuild — but only touches
    // one slot. The GPU culling buffer layout is unchanged.
    // When using the UploadManager path, `onComplete` is invoked after the
    // transfer fence signals (async). For the legacy staging path, it's
    // called when the pending transfer fence signals.
    // Returns true on success.
    bool uploadSlot(VulkanApp* app, uint32_t slotIndex, float priority = 0.0f,
                    std::function<void()> onComplete = nullptr);

    // Upload a single mesh to GPU (incremental update). Requires buffers to have capacity.
    // Returns true if upload succeeded, false if rebuild() is needed (capacity exceeded or buffers not created).
    // Setters for async buffer publication (called when an async upload finishes)
    void setVertexBufferForMesh(uint32_t meshId, Buffer vbuf);
    void setIndexBufferForMesh(uint32_t meshId, Buffer ibuf);
    
    // Erase a mesh from GPU by zeroing its indirect command (prevents culling from reading trash).

        public:
            // Needed for main.cpp and other modules
    // Call after removeMesh() for runtime removals.
    void eraseMeshFromGPU(VulkanApp* app, uint32_t meshId);
    
    // Set which per-frame cull buffers to use. Must be called once per frame
    // before prepareCull / drawPrepared. frame idx should be in [0, MAX_CULL_FRAMES).
    void setCullFrame(uint32_t frame);
    
    // Ensure GPU buffers have capacity for at least the given counts. 
    // Call this before a batch of addMesh+uploadMesh if you know the expected size.
    // Returns true if buffers are ready, false if they needed to be created/grown (triggers rebuild).
    bool ensureCapacity(size_t vertexCount, size_t indexCount, size_t meshCount);
    
    // Check if dirty flag is set (needs rebuild or incremental uploads)
    bool isDirty() const { return dirty; }

    // Returns true when the GPU indirect/bounds buffers have never been written
    // (metaBuffersWrittenCount == 0) but active meshes exist — the GPU buffers
    // still contain stale data from a previous scene.  Callers should force a
    // full rebuild instead of the incremental path.
    bool needsFullRebuild() const {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        return metaBuffersWrittenCount == 0 && !meshes.empty();
    }

public:

    // Poll for completion of an in-flight async transfer and publish
    // the results (update meta-buffers, etc.).  Call once per frame
    // before acquireBuffers so deferred publications are visible to
    // the current frame's draws.
    void pollPendingTransfers(VulkanApp* app);

    // Acquire vertex/index buffers from the transfer queue. Must be called once
    // per frame before draws. Records a buffer memory barrier with no QFO
    // (VK_QUEUE_FAMILY_IGNORED — buffers are CONCURRENT) to make transfer
    // writes visible to vertex/index input stages.
    void acquireBuffers(VkCommandBuffer cmd);

    // Run GPU culling/compaction (must be called outside any render pass).
    // `camPos`/`lodBias` drive the per-chunk LoD band selection.
    void prepareCull(VkCommandBuffer cmd, const glm::mat4& viewProj,
                     glm::vec3 camPos = glm::vec3(0.0f), float lodBias = 8.0f);
    // Run GPU culling into caller-provided output buffers using a provided compute descriptor set.
    void prepareCullWithDescriptor(VkCommandBuffer cmd, const glm::mat4& viewProj, VkDescriptorSet computeDesc,
                                   VkBuffer outCompactBuffer, VkBuffer outVisibleCountBuffer,
                                   glm::vec3 camPos = glm::vec3(0.0f), float lodBias = 8.0f);
    // Issue indirect draw using the compacted indirect buffer (call inside render pass).
    void drawPrepared(VkCommandBuffer cmd, uint32_t maxDraws = 0);
    void drawPreparedWithBuffers(VkCommandBuffer cmd, VkBuffer compactBuffer, VkBuffer visibleCountBuffer, uint32_t maxDraws = 0);
    void bindBuffers(VkCommandBuffer cmd);
    void drawIndirectOnly(VkCommandBuffer cmd, VulkanApp* app, uint32_t maxDraws = 0);
    void drawIndirectOnly(VkCommandBuffer cmd, VkPipelineLayout pipelineLayout, uint32_t maxDraws = 0);

    // ── Cascade-aware culling (single pass, 3 cascades) ──
    // Single compute dispatch that culls all chunks against 3 cascade frustums
    // simultaneously. Each chunk is culled independently per cascade.
    // `camPos`/`lodBias` mirror the main pass so shadow draws use the same
    // per-chunk LoD selection.
    void prepareCullCascades(VkCommandBuffer cmd,
                             const glm::mat4 cascadeMatrices[3],
                             glm::vec3 camPos = glm::vec3(0.0f), float lodBias = 8.0f);
    // Draw a specific cascade's compacted output (call inside render pass).
    void drawCascadeOnly(VkCommandBuffer cmd, uint32_t cascadeIndex);

    // Accessors
    const Buffer& getIndirectBuffer() const { return indirectBuffer; }
    const Buffer& getBoundsBuffer() const { return boundsBuffer; }
    VkDescriptorSetLayout getComputeDescriptorSetLayout() const { return computeDescriptorSetLayout; }

    // Get the pre-allocated capacity (indirect command count / max slots)
    // In slotted mode this is the fixed slot pool size; in legacy mode it grows
    // with addMesh(). Used for sizing external compact buffers (e.g. cubemap).
    size_t getMeshCapacity() const { return meshCapacity; }

    // Persistent scratch buffer bound to binding 4 of the cull compute layout
    // by external descriptor-set owners (cubemap faces, async backface pass).
    VkBuffer getVisibleLodsScratchBuffer() const { return visibleLodsScratch.buffer; }

    // Get count of active meshes (memoized: recomputed under the same mutex
    // only after a meshes mutation, so per-frame stats/sizing calls do not
    // scan the whole map).
    size_t getMeshCount() const {
        std::lock_guard<std::recursive_mutex> lock(mutex);
        return activeMeshCountLocked();
    }
    // Total merged vertex/index counts (used for capacity planning)
    size_t getMergedVertexCount() const;
    size_t getMergedIndexCount() const;

    // Host-read of the GPU-visible count. Uses a per-frame fence to avoid
    // stalling unrelated queue work.
    uint32_t readVisibleCount(VulkanApp* app) const;

    // Query mesh info (copy) for use in the app (bounds, offsets, flags).
    MeshInfo getMeshInfo(uint32_t meshId) const;

    // Invoke `visitor(const MeshInfo&)` for each active mesh (thread-safe).
    // Avoids allocating a temporary vector.
    template<typename F>
    void visitActiveMeshInfos(F&& visitor) const {
        std::lock_guard<std::recursive_mutex> guard(mutex);
        for (const auto& kv : meshes) {
            if (kv.second.active) std::forward<F>(visitor)(kv.second);
        }
    }

private:
    struct PendingTransfer {
        VkFence fence = VK_NULL_HANDLE;
        // Staging region suballocated from the app's persistent StagingRingBuffer
        // (preferred). When the ring cannot satisfy the request we fall back to a
        // dedicated staging buffer stored in `stagingBuffer`.
        StagingRingBuffer::Allocation stagingAlloc = {};
        Buffer stagingBuffer = {};
    };
    PendingTransfer pendingTransfer = {};

    mutable std::recursive_mutex mutex;
    void publishPendingTransfer(VulkanApp* app);
    // Unlocked variant — caller must hold mutex.
    void doUploadMeshMetaBuffers(VulkanApp* app);
    // Publish ONE mesh's indirect command + bounds at its current drawIndex.
    // Unlocked — caller must hold mutex. Used by the async UploadManager path
    // where transfers may complete out of order, so the contiguous
    // append-only watermark (doUploadMeshMetaBuffers) cannot be used.
    void publishMeshMeta(uint32_t meshId);
    // Unlocked — caller must hold `mutex`. Memoized active-mesh count;
    // recomputed (full scan) only after any meshes mutation.
    size_t activeMeshCountLocked() const;
    // Unlocked — caller must hold `mutex`. Number of draw commands (slots)
    // to cull: fixed slot pool capacity in slotted mode, active mesh count
    // in legacy mode.
    uint32_t getCullDispatchCountLocked() const;

    // ── Slotted-mode internals ──
    // Copy vertex/index data from a Geometry into the level's packed span in
    // mergedVertices/mergedIndices (absolute offsets). Requires slotted mode.
    void copyGeometryToLevel(const Geometry& mesh, MeshInfo::LevelData& ld);

    // Async transfer engine (optional). When non-null, uploadMeshes routes
    // through it instead of the single-slot pendingTransfer path.
    streaming::UploadManager* uploadMgr_ = nullptr;
    streaming::StreamCategory streamCategory_ = streaming::StreamCategory::Solid;

    // Deferred upload completion callbacks for the legacy staging path.
    // Fired when publishPendingTransfer detects the staging fence has signaled.
    // The UploadManager path uses job.onComplete instead.
    std::vector<std::function<void()>> deferredUploadCallbacks_;
    uint32_t nextId = 1;
    std::unordered_map<uint32_t, MeshInfo> meshes; // chunkId -> MeshInfo
    // Memoized active-mesh count backing getMeshCount()/dispatch-count reads.
    // Invalidated by every mutation of `meshes` (all under `mutex`), so the
    // cached value always equals a fresh full scan.
    mutable bool activeMeshCountDirty_ = true;
    mutable size_t activeMeshCount_ = 0;

    // CPU-side combined buffers
    std::vector<Vertex> mergedVertices;
    std::vector<uint32_t> mergedIndices;
    std::vector<VkDrawIndexedIndirectCommand> indirectCommands;

    // When true, the slotted API is active. In this mode, mergedVertices and
    // mergedIndices are pre-sized to the shared element pool capacity and each
    // chunk owns a draw-entry block (see slotAlloc). The indirectCommands
    // vector is also pre-sized. No full rebuilds are performed; each block and
    // level span is updated independently.
    bool slottedMode = false;

    // Draw-entry allocator: one draw entry per active chunk. allocate(1,1)/
    // free under a capacity of 1 — the allocated index IS the "slot"
    // (nest-level identity) and the draw entry index.
    SlotAllocator slotAlloc;

    // Packed element pools: variable-size spans of vertex/index elements
    // shared by all chunks (a chunk's levels are independent spans).
    PackedSpaceAllocator spaceAlloc;

    // Last slot-usage high-water mark logged (DEBUG capacity tuning aid).
    uint32_t lastPeakLogged_ = 0;

    // Set which per-frame cull buffers to use. Must be called once per frame
    // before prepareCull / drawPrepared. frame idx should be in [0, MAX_CULL_FRAMES).

    // A temporary compact indirect buffer used to upload only visible commands — per-frame to avoid cross-frame races
    std::array<Buffer, MAX_CULL_FRAMES> compactIndirectBuffers;
    // Per-frame chosen-LoD output from the cull compute shader (uvec2 per kept
    // entry: drawIndex, chosen level). Written by binding 4 of the cull
    // pipeline; zeroed together with the compact buffer each prepareCull.
    std::array<Buffer, MAX_CULL_FRAMES> visibleLodBuffers;
    // Dedicated visibleLods buffer for the caller-provided-descriptor paths
    // (cubemap faces, async backface): those sets bind this scratch buffer so
    // their dispatches never race the per-frame zero fills.
    Buffer visibleLodsScratch;
    // GPU-side culling resources
    Buffer boundsBuffer; // vec4 per draw entry: min, max, meta{cellSize, level, maxLevel, unused}
    // Per-frame visible count buffers
    std::array<Buffer, MAX_CULL_FRAMES> visibleCountBuffers;
    // Persistent host mapping for zeroing visible counts (avoids vkCmdFillBuffer + barrier on RADV)
    mutable std::array<uint32_t*, MAX_CULL_FRAMES> visibleCountMapped = {nullptr, nullptr, nullptr};
    VkDevice storedDevice = VK_NULL_HANDLE;

    // Compute pipeline objects for GPU culling
    TrackedHandle<VkPipeline> computePipeline;
    TrackedHandle<VkPipelineLayout> computePipelineLayout;
    TrackedHandle<VkDescriptorSetLayout> computeDescriptorSetLayout;
    TrackedHandle<VkDescriptorPool> computeDescriptorPool;
    std::array<TrackedHandle<VkDescriptorSet>, MAX_CULL_FRAMES> computeDescriptorSets;

    // ── Cascade-aware culling (per-frame resources) ──
    struct CascadeCullFrame {
        std::array<Buffer, 3> compactBuffers; // compact output per cascade
        std::array<Buffer, 3> countBuffers;   // visible count per cascade
        std::array<uint32_t*, 3> countMapped = {nullptr, nullptr, nullptr};
        TrackedHandle<VkDescriptorSet> descSet;
    };
    std::array<CascadeCullFrame, MAX_CULL_FRAMES> cascadeCullFrames;
    Buffer cascadeMatrixBuffer; // storage buffer with 3 mat4 cascade matrices
    TrackedHandle<VkPipeline> cascadeCullPipeline;
    TrackedHandle<VkPipelineLayout> cascadeCullPipelineLayout;
    TrackedHandle<VkDescriptorSetLayout> cascadeCullDescSetLayout;
    TrackedHandle<VkDescriptorPool> cascadeCullDescPool;
    bool cascadeCullInited = false;
    VkBuffer cascadeDescIndirectBuffer = VK_NULL_HANDLE; // tracks which indirectBuffer the descriptors reference
    VkBuffer cascadeDescBoundsBuffer = VK_NULL_HANDLE;   // tracks which boundsBuffer the descriptors reference
    VulkanApp* cascadeDescApp = nullptr; // stored for descriptor refresh
    void initCascadeCull(VulkanApp* app);
    void destroyCascadeCull();
    void updateCascadeDescriptor(VulkanApp* app, uint32_t frame);
    void refreshCascadeDescriptorsIfNeeded();

    // Optional device function for indirect-count draw (KHR or core 1.2)
    PFN_vkCmdDrawIndexedIndirectCountKHR cmdDrawIndexedIndirectCount = nullptr;

    // GPU buffers
    // Geometry (vertex/index) is double-buffered across a small fixed pool of
    // slots. rebuild() uploads a fresh full copy into a slot that no in-flight
    // frame is reading, then swaps the "current" slot for subsequent draws — so
    // the brush flow no longer needs a device-wide deviceWaitIdle() to avoid a
    // WRITE_AFTER_READ hazard. Slots are created once and reused (grown only
    // when capacity increases — no per-rebuild alloc churn). A slot is recycled
    // (marked free) via a frame-fence-gated, NON-blocking deferred callback
    // (deferDestroyUntilFence), so there is no vkWaitForFences in the rebuild
    // path and thus no fence-index wraparound deadlock. Pool size =
    // MAX_FRAMES_IN_FLIGHT (3) + 3 headroom so a free slot is virtually always
    // available under 1–2 rebuilds/frame; a temporary throwaway allocation is
    // used as a bounded fallback if none is free.
    static constexpr uint32_t MAX_GEOM_BUFFERS = 6;
    std::array<Buffer, MAX_GEOM_BUFFERS> vertexSlots{};
    std::array<Buffer, MAX_GEOM_BUFFERS> indexSlots{};
    std::array<size_t, MAX_GEOM_BUFFERS> vertexSlotCap{};
    std::array<size_t, MAX_GEOM_BUFFERS> indexSlotCap{};
    std::array<bool, MAX_GEOM_BUFFERS> geomSlotInUse{};
    uint32_t currentGeomSlot = UINT32_MAX;
    // Mirrors of the current slot's buffers. All bind/draw/barrier and
    // incremental-upload paths reference these; kept in sync on every slot swap.
    Buffer vertexBuffer;
    Buffer indexBuffer;
    // Reserve a free geometry slot (marks it in-use). Caller must hold `mutex`.
    // Returns UINT32_MAX when the pool is exhausted (caller uses fallback path).
    uint32_t acquireGeomSlot();
    // Mark a slot reusable. Locks `mutex` — invoked from the deferred-destroy
    // processor, not from within rebuild().
    void markGeomSlotFree(uint32_t slot);
    // Recycle the geometry buffers that were current before a rebuild swap:
    // pool slots are returned to the free list, throwaway fallback buffers are
    // destroyed — both gated on the current frame fence. Caller must hold `mutex`.
    void recyclePreviousGeom(VulkanApp* app, uint32_t prevSlot, Buffer prevVertex, Buffer prevIndex);
    Buffer indirectBuffer;
    
    // Capacity tracking (in elements, not bytes)
    size_t vertexCapacity = 0;
    size_t indexCapacity = 0;
    size_t meshCapacity = 0;

    // Tracks how many active mesh entries have been written to GPU
    // indirect/bounds buffers. Used for append-only writes to avoid
    // rewriting existing entries while in-flight GPU frames read them.
    size_t metaBuffersWrittenCount = 0;

    bool dirty = false;
    uint32_t currentCullFrame = 0;
    bool descriptorDirty = false;  // flag for deferred descriptor update
    VkDescriptorSet pendingDescriptorSet = VK_NULL_HANDLE; // ds to update (VK_NULL_HANDLE means use/create material set)

    // ── Staged meta writes ───────────────────────────────────────────────────
    // Host-side publishes (draw command + bounds) must NEVER memcpy directly
    // into the HOST_VISIBLE indirect/bounds buffers: an in-flight cull
    // dispatch may be reading the same entry mid-write, tearing the 20-byte
    // DrawCmd and producing a garbage indexCount that makes
    // vkCmdDrawIndexedIndirectCount spin the GE (GPU hang observed on RADV /
    // Radeon 680M — see the compact-buffer zero fills). Instead the writes are
    // queued here and copied to the GPU buffers via vkCmdCopyBuffer at the
    // start of the next prepareCull (main thread, same command buffer), which
    // is queue-ordered after every in-flight frame's reads.
    struct MetaStageRecord {
        uint32_t entryIndex = 0;
        VkDrawIndexedIndirectCommand cmd{};
        glm::vec4 bounds[3] = { glm::vec4(0.0f), glm::vec4(0.0f), glm::vec4(0.0f) };
        bool boundsValid = false;
    };
    std::array<std::vector<MetaStageRecord>, MAX_CULL_FRAMES> metaStagePending_;
    std::vector<MetaStageRecord> metaStageFlush_;
    std::array<Buffer, MAX_CULL_FRAMES> metaStageBuffers{};
    std::array<uint32_t, MAX_CULL_FRAMES> metaStageCapBytes{};
    VulkanApp* app_ = nullptr;
    // Caller must hold `mutex`. Queues a host-side write for flushStagedMetaWrites.
    void stageMeshMetaWrite(uint32_t entryIndex, const VkDrawIndexedIndirectCommand& cmd,
                            const glm::vec4* bounds, bool boundsValid);
    // Copies the frame's staged writes into the GPU indirect/bounds buffers via
    // vkCmdCopyBuffer (queue-ordered, race-free). Called at the top of
    // prepareCull on the main thread before any acquireBuffers barrier.
    void flushStagedMetaWrites(VkCommandBuffer cmd, uint32_t frame);
};
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
        // Shared column anchor for the LoD band gate: the FINEST chunk's min
        // corner. All rungs of a column share this so the clipmap anchor nests
        // and exactly one rung is selected per region (no overlap).
        glm::vec4 boundsBase = glm::vec4(0.0f);
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
                            int level = 0, const glm::vec3* boundsBase = nullptr);
    void removeMeshSlotted(uint32_t slotIndex);

    // Maximum LoD ladder level present in the tree. Written into each entry's
    // bounds meta as `lodMeta.z` and used by the GPU distance-band test to clamp
    // the selected level. MUST equal the tree's real ladder depth
    // (LocalScene::maxChunkLod) — hardcoding it (e.g. 4) permanently culls every
    // chunk whose level exceeds it, leaving holes across the terrain.
    void setMaxLodLevel(int l) { maxLodLevel_ = l; }

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

    // Re-point the core compute descriptor-set bindings (0..4: inCmds, outCmds,
    // bounds, visibleCount, visibleLods) to the CURRENT buffer handles for frame
    // `f`. rebuild() can recreate these buffers (e.g. when meshCapacity grows), and
    // the once-written descriptor sets would otherwise keep pointing at freed
    // handles, so the cull reads stale/empty data and emits zero visible draws.
    // Called every frame from recordCull before the dispatch.
    void updateCoreComputeDescriptors(uint32_t f);
    
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
                     glm::vec3 camPos = glm::vec3(0.0f), float lodBias = 8.0f, int maxTargetLod = 16,
                     const glm::mat4* cascadeMatrices = nullptr, bool doCascade = false, bool doMain = true,
                     bool doVegCascade = false, uint32_t vegChunkCount = 0);
    // Run GPU culling into caller-provided output buffers using a provided compute descriptor set.
    void prepareCullWithDescriptor(VkCommandBuffer cmd, const glm::mat4& viewProj, VkDescriptorSet computeDesc,
                                    VkBuffer outCompactBuffer, VkBuffer outVisibleCountBuffer,
                                    glm::vec3 camPos = glm::vec3(0.0f), float lodBias = 8.0f, int maxTargetLod = 16);

    // ── SDF debug-cube culling (merged into the solid indirect.comp dispatch) ──
    // Supplies the AABBs of the SDF debug cubes. prepareCull appends them after
    // the solid entries and frustum-culls them in the SAME indirect.comp dispatch,
    // writing survivors to a dedicated SDF output stream. The SDF debug renderer
    // then draws from those buffers (see getSdfCompactBuffer / getSdfCountBuffer).
    struct SdfCube {
        glm::vec3 minp;
        glm::vec3 maxp;
        // LoD meta so the SDF cull can apply the SAME clipmap band gate as the solid
        // terrain (keeps exactly one rung per region, no overlap with finer rungs).
        // Mirrors the solid bounds entry: cellSize = chunk cube side, level = chunkLod
        // rung, base = chunk min corner (shared column anchor).
        float cellSize = 0.0f;
        int level = 0;
        glm::vec3 base = glm::vec3(0.0f);
    };
    void setSdfCubes(const std::vector<SdfCube>& cubes);
    // Capacity of the folded SDF command stream (sdfCompactBuf): the maximum number
    // of SDF DrawCmds indirect.comp can emit per frame. Used to bound maxDrawCount.
    uint32_t getMaxSdfCommands() const { return MAX_SDF_CUBES; }
    VkBuffer getSdfCompactBuffer(uint32_t frame) const {
        return frame < MAX_CULL_FRAMES ? sdfCompactBuf[frame].buffer : VK_NULL_HANDLE;
    }
    VkBuffer getSdfCountBuffer(uint32_t frame) const {
        return frame < MAX_CULL_FRAMES ? sdfCountBuf[frame].buffer : VK_NULL_HANDLE;
    }

    // ── Mesh bounding-box culling (merged into the solid indirect.comp dispatch) ──
    // Supplies the AABBs of the meshes currently uploaded to the GPU. prepareCull
    // appends them after the solid + SDF entries and frustum-culls them in the SAME
    // indirect.comp dispatch, writing survivors to a dedicated bbox output stream.
    // The bounding-box debug renderer then draws from those buffers (see
    // getBboxCompactBuffer / getBboxCountBuffer).
    struct BBox {
        glm::vec3 minp;
        glm::vec3 maxp;
        // LoD meta so the bbox cull can apply the SAME clipmap band gate as the
        // solid terrain (keeps exactly one rung per region, no overlap). Mirrors
        // the solid bounds entry: cellSize = chunk cube side, level = chunkLod
        // rung, base = chunk min corner (shared column anchor).
        float cellSize = 0.0f;
        int level = 0;
        glm::vec3 base = glm::vec3(0.0f);
    };
    void setBoundingBoxes(const std::vector<BBox>& boxes);
    // Capacity of the folded bounding-box command stream (bboxCompactBuf): the max
    // number of bbox DrawCmds indirect.comp can emit per frame. Used to bound maxDrawCount.
    uint32_t getMaxBboxCommands() const { return MAX_BBOX_CUBES; }
    VkBuffer getBboxCompactBuffer(uint32_t frame) const {
        return frame < MAX_CULL_FRAMES ? bboxCompactBuf[frame].buffer : VK_NULL_HANDLE;
    }
    VkBuffer getBboxCountBuffer(uint32_t frame) const {
        return frame < MAX_CULL_FRAMES ? bboxCountBuf[frame].buffer : VK_NULL_HANDLE;
    }

    // ── Vegetation cull integration ──
    // The solid IndirectRenderer owns the merged indirect.comp dispatch, which
    // emits BOTH the solid terrain commands AND (for every visible solid chunk
    // that carries vegetation) the billboard/impostor commands. The four
    // compact/count output buffers are OWNED by VegetationRenderer and simply
    // (re)bound here each frame; its existing draw paths then read the merged
    // dispatch's output. Per-frame registration:
    //   setVegetationCullData(bbCompact, bbCount, impCompact, impCount)
    // The per-solid-draw vegetation table (binding 9) is built internally by the
    // IndirectRenderer from a {nid -> {instanceCount, firstInstance}} map supplied
    // by VegetationRenderer via setVegetationChunkInfo().
    void setVegetationCullData(const std::array<Buffer, MAX_CULL_FRAMES>& bbCompact,
                               const std::array<Buffer, MAX_CULL_FRAMES>& bbCount,
                               const std::array<Buffer, MAX_CULL_FRAMES>& impCompact,
                               const std::array<Buffer, MAX_CULL_FRAMES>& impCount);
    // Per-solid-chunk vegetation metadata, keyed by the solid mesh id (== the
    // uint32 meshId used as the draw entry index). value = vec4(instanceCount,
    // firstInstance, 0, 0).
    void setVegetationChunkInfo(const std::unordered_map<uint32_t, glm::vec4>& info);
    // Vegetation cascade (shadow) streams: bind the per-cascade billboard/impostor
    // command + count buffers into the merged indirect.comp descriptor set (bindings
    // 24..36). Called by VegetationRenderer each frame so re-grows are reflected and
    // the veg cascade dispatch (doVegCascade=true) writes into VegetationRenderer's
    // own output buffers. VEG_CULL_FRAMES == MAX_CULL_FRAMES == 3.
    void setVegCascadeData(VkBuffer chunkInfo,
                           const std::array<std::array<VkBuffer, 3>, MAX_CULL_FRAMES>& bbCompact,
                           const std::array<std::array<VkBuffer, 3>, MAX_CULL_FRAMES>& bbCount,
                           const std::array<std::array<VkBuffer, 3>, MAX_CULL_FRAMES>& impCompact,
                           const std::array<std::array<VkBuffer, 3>, MAX_CULL_FRAMES>& impCount);
    // Rebuild the per-draw vegetation table (binding 9) from vegChunkInfoMap.
    // Called each frame from prepareCull so it tracks the slotted renderer's
    // incremental draw-index updates.
    void updateVegTable();
    // Output buffers produced by the merged dispatch (billboards / impostors),
    // consumed by VegetationRenderer::draw. Valid after prepareCull for `frame`.
    VkBuffer getVegBbCompact(uint32_t frame) const { return vegBbCompactBuf[frame]; }
    VkBuffer getVegBbCount(uint32_t frame) const  { return vegBbCountBuf[frame]; }
    VkBuffer getVegImpCompact(uint32_t frame) const { return vegImpCompactBuf[frame]; }
    VkBuffer getVegImpCount(uint32_t frame) const  { return vegImpCountBuf[frame]; }
    VkBuffer getVegTableBuffer() const { return vegTableBuffer.buffer; }
    bool isVegetationCullEnabled() const { return vegCullEnabled; }
    // Dummy buffer bound to the vegetation bindings (5..9) on dispatches that do
    // not run vegetation culling, so the layout is always complete.
    VkBuffer getVegDummyBuffer() const { return vegDummyBuffer.buffer; }
    // Current cull frame index (so callers can align their draw frame with the one
    // the merged dispatch wrote into).
    uint32_t getCurrentCullFrame() const { return currentCullFrame; }
    // Issue indirect draw using the compacted indirect buffer (call inside render pass).
    void drawPrepared(VkCommandBuffer cmd, uint32_t maxDraws = 0);
    void drawPreparedWithBuffers(VkCommandBuffer cmd, VkBuffer compactBuffer, VkBuffer visibleCountBuffer, uint32_t maxDraws = 0);
    void bindBuffers(VkCommandBuffer cmd);
    void drawIndirectOnly(VkCommandBuffer cmd, VulkanApp* app, uint32_t maxDraws = 0);
    void drawIndirectOnly(VkCommandBuffer cmd, VkPipelineLayout pipelineLayout, uint32_t maxDraws = 0);

    // ── Cascade-aware culling (single pass, 3 cascades) ──
    // Single compute dispatch that culls all chunks against 3 cascade frustums
    // simultaneously. Each chunk is culled independently per cascade.
    // Cascade streams are now emitted by the merged indirect.comp dispatch
    // (prepareCull with doCascade=true), not a separate pipeline.
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
    // Real ladder depth of the tree (set via setMaxLodLevel from
    // LocalScene::maxChunkLod). Used as lodMeta.z in the GPU band test.
    int maxLodLevel_ = 16;

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
    // ── Vegetation cull integration (merged single dispatch) ──
    // The merged indirect.comp dispatch writes the vegetation billboard/impostor
    // commands into the buffers OWNED by VegetationRenderer (so its existing draw
    // paths are unchanged). We only hold the bound VkBuffer handles here, supplied
    // via setVegetationCullData each frame. Billboards use indexCount=36,
    // impostors indexCount=6.
    std::array<VkBuffer, MAX_CULL_FRAMES> vegBbCompactBuf = {};  // binding 7
    std::array<VkBuffer, MAX_CULL_FRAMES> vegBbCountBuf   = {};  // binding 8
    std::array<VkBuffer, MAX_CULL_FRAMES> vegImpCompactBuf = {}; // binding 5
    std::array<VkBuffer, MAX_CULL_FRAMES> vegImpCountBuf  = {};  // binding 6
    // Per-solid-draw vegetation table (binding 9 input): vec4{instanceCount,
    // firstInstance, 0, 0} indexed by the solid draw entry index s. Built by the
    // IndirectRenderer from a {nid -> vec4} map supplied via setVegetationChunkInfo
    // (VegetationRenderer owns the per-chunk instance counts/offsets). The AABB used
    // for veg is the solid chunk's AABB (the shader reuses boundsBuf[s]).
    Buffer vegTableBuffer;
    void* vegTableMapped = nullptr;
    uint32_t vegTableCapacity = 0;

    // ── SDF debug-cube culling (folded into the solid indirect.comp dispatch) ──
    // Per cull-frame buffers for the SDF stream. SDF cubes are appended after the
    // solid entries in the dispatch and written to sdfCompactBuf (binding 10) with
    // their count in sdfCountBuf (binding 11). Inputs are sdfInCmdsBuf (binding 12)
    // and sdfBoundsBuf (binding 13). Capacity must hold every chunkLod==1 surface
    // cube emitted by LocalScene::requestSDFCubes (can be hundreds of thousands for a
    // large terrain), so it is sized generously; the per-instance data lives in the
    // DebugSDFRenderer instance buffers (grown by ensureCullCapacity to maxStorageBufferRange).
    static constexpr uint32_t MAX_SDF_CUBES = 500000;
    std::array<Buffer, MAX_CULL_FRAMES> sdfCompactBuf;
    std::array<Buffer, MAX_CULL_FRAMES> sdfCountBuf;
    std::array<Buffer, MAX_CULL_FRAMES> sdfInCmdsBuf;
    std::array<Buffer, MAX_CULL_FRAMES> sdfBoundsBuf;
    std::vector<SdfCube> sdfCubes_;

    // ── Mesh bounding-box culling (folded into the solid indirect.comp dispatch) ──
    // Inputs (bboxBoundsBuf) are host-written each frame from the mesh AABBs and
    // read by the cull; outputs (bboxCompactBuf/bboxCountBuf) are GPU-written and
    // consumed by the bounding-box indirect draw. Capacity bounds the worst case
    // (one box per uploaded mesh).
    static constexpr uint32_t MAX_BBOX_CUBES = 500000;
    std::array<Buffer, MAX_CULL_FRAMES> bboxCompactBuf;
    std::array<Buffer, MAX_CULL_FRAMES> bboxCountBuf;
    std::array<Buffer, MAX_CULL_FRAMES> bboxBoundsBuf;
    std::vector<BBox> bboxCubes_;
    std::unordered_map<uint32_t, glm::vec4> vegChunkInfoMap;
    bool vegCullEnabled = false;
    // Dummy bound to the veg bindings (5..9) on the solid-only dispatch — the
    // indirect.comp shader statically references them so they must always be
    // valid, even when vegetation isn't being culled.
    Buffer vegDummyBuffer;
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
    Buffer cascadeDummyBuffer;  // bound to the cascade bindings of external (Solid360) descriptor sets
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

    // Vegetation cascade (shadow) streams, bound into the merged indirect.comp
    // descriptor set (bindings 24..36) so a single dispatch can also emit the
    // vegetation cascade billboard + impostor command streams. Bound once by
    // VegetationRenderer after both renderers are initialized (VEG_CULL_FRAMES ==
    // MAX_CULL_FRAMES == 3, so the per-frame indices line up).
    VkBuffer vegCascadeInfoBuffer = VK_NULL_HANDLE;
    std::array<std::array<VkBuffer, 3>, MAX_CULL_FRAMES> vegCascadeBbCompact{};
    std::array<std::array<VkBuffer, 3>, MAX_CULL_FRAMES> vegCascadeBbCount{};
    std::array<std::array<VkBuffer, 3>, MAX_CULL_FRAMES> vegCascadeImpCompact{};
    std::array<std::array<VkBuffer, 3>, MAX_CULL_FRAMES> vegCascadeImpCount{};
    bool vegCascadeInited = false;

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
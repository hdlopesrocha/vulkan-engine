// IndirectRenderer.hpp
#pragma once

#include "../VulkanApp.hpp"
#include "../TrackedHandle.hpp"
#include "../../math/Geometry.hpp"
#include <glm/glm.hpp>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <shared_mutex>
#include <cstdint>

#include <array>
#include "CommandBufferState.hpp"
#include "../streaming/StreamCommon.hpp"

namespace streaming { class UploadManager; }

// Manages a single large vertex/index/indirect buffer and provides a simple
// CPU-side allocator for adding/removing meshes. Draws are performed via
// vkCmdDrawIndexedIndirect (one indirect command per mesh). The allocator is
// append-first and supports reclamation on remove (simple free list rebuild).
class IndirectRenderer {
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
        bool active = false;
        // NOTE: per-mesh buffers removed — meshes are packed into the merged buffers
    };

    IndirectRenderer();
    ~IndirectRenderer();

    void init();
    void cleanup();

    // Add mesh and return mesh id. Mesh transform is identity on GPU (no per-mesh model SSBO/push-constants).
    uint32_t addMesh(const Geometry& mesh);
    // Add mesh with a custom ID (e.g., node ID from octree). If mesh with this ID exists, it is replaced.
    uint32_t updateMesh(const Geometry& mesh, uint32_t customId);
    void removeMesh(uint32_t meshId);
    // Remove all meshes and reset GPU write tracking.  Call before reloading a
    // scene so new meshes are written from position 0 in the indirect/bounds
    // buffers — without this, stale entries from the previous scene cause the
    // culling compute shader to read garbage, resulting in GPU hangs.
    void removeAllMeshes();

    // Rebuild GPU backing buffers from current CPU mesh list. Call before drawing
    // if add/remove operations occurred.
    void rebuild(VulkanApp* app);

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
        std::shared_lock<std::shared_mutex> lock(mutex);
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
    void prepareCull(VkCommandBuffer cmd, const glm::mat4& viewProj, uint32_t maxDraws = 0);
    // Run GPU culling into caller-provided output buffers using a provided compute descriptor set.
    void prepareCullWithDescriptor(VkCommandBuffer cmd, const glm::mat4& viewProj, VkDescriptorSet computeDesc,
                                   VkBuffer outCompactBuffer, VkBuffer outVisibleCountBuffer, uint32_t maxDraws = 0);
    // Issue indirect draw using the compacted indirect buffer (call inside render pass).
    void drawPrepared(VkCommandBuffer cmd, uint32_t maxDraws = 0);
    void drawPreparedWithBuffers(VkCommandBuffer cmd, VkBuffer compactBuffer, VkBuffer visibleCountBuffer, uint32_t maxDraws = 0);
    void bindBuffers(VkCommandBuffer cmd);
    void drawIndirectOnly(VkCommandBuffer cmd, VulkanApp* app, uint32_t maxDraws = 0);
    void drawIndirectOnly(VkCommandBuffer cmd, VkPipelineLayout pipelineLayout, uint32_t maxDraws = 0);

    // Accessors
    const Buffer& getVertexBuffer() const { return vertexBuffer; }
    const Buffer& getIndexBuffer() const { return indexBuffer; }
    const Buffer& getIndirectBuffer() const { return indirectBuffer; }
    const Buffer& getCompactIndirectBuffer() const { return compactIndirectBuffers[currentCullFrame]; }
    const Buffer& getBoundsBuffer() const { return boundsBuffer; }
    VkPipeline getComputePipeline() const { return computePipeline; }
    VkDescriptorSetLayout getComputeDescriptorSetLayout() const { return computeDescriptorSetLayout; }
    VkDescriptorPool getComputeDescriptorPool() const { return computeDescriptorPool; }

    // Get count of active meshes
    size_t getMeshCount() const {
        std::shared_lock<std::shared_mutex> lock(mutex);
        size_t count = 0;
        for (const auto& m : meshes) {
            if (m.second.active) ++count;
        }
        return count;
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
        std::shared_lock<std::shared_mutex> guard(mutex);
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

    mutable std::shared_mutex mutex;
    void publishPendingTransfer(VulkanApp* app);
    // Unlocked variant — caller must hold mutex.
    void doUploadMeshMetaBuffers(VulkanApp* app);
    // Publish ONE mesh's indirect command + bounds at its current drawIndex.
    // Unlocked — caller must hold mutex. Used by the async UploadManager path
    // where transfers may complete out of order, so the contiguous
    // append-only watermark (doUploadMeshMetaBuffers) cannot be used.
    void publishMeshMeta(uint32_t meshId);

    // Async transfer engine (optional). When non-null, uploadMeshes routes
    // through it instead of the single-slot pendingTransfer path.
    streaming::UploadManager* uploadMgr_ = nullptr;
    streaming::StreamCategory streamCategory_ = streaming::StreamCategory::Solid;
    uint32_t nextId = 1;
    std::unordered_map<uint32_t, MeshInfo> meshes; // nodeId -> MeshInfo

    // CPU-side combined buffers
    std::vector<Vertex> mergedVertices;
    std::vector<uint32_t> mergedIndices;
    std::vector<VkDrawIndexedIndirectCommand> indirectCommands;

    // Set which per-frame cull buffers to use. Must be called once per frame
    // before prepareCull / drawPrepared. frame idx should be in [0, MAX_CULL_FRAMES).

    // A temporary compact indirect buffer used to upload only visible commands — per-frame to avoid cross-frame races
    std::array<Buffer, MAX_CULL_FRAMES> compactIndirectBuffers;
    // GPU-side culling resources
    Buffer boundsBuffer; // vec4 per-mesh: xyz=center, w=radius
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
public:
    CommandBufferState* cmdState = nullptr;
    void setCmdState(CommandBufferState* state) { cmdState = state; }
};
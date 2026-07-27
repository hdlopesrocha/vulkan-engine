#pragma once

#include "Chunk.hpp"
#include "../vulkan/renderer/ChunkManager.hpp"
#include "../vulkan/renderer/RenderProxy.hpp"
#include "../utils/LocalScene.hpp"
#include "../utils/SolidSpaceChangeHandler.hpp"
#include "../utils/LiquidSpaceChangeHandler.hpp"
#include "../space/OctreeChangeHandler.hpp"
#include <unordered_map>
#include <memory>
#include <mutex>
#include <functional>

// The World owns all terrain data and chunk state.
//
// Responsibility boundary:
//   WORLD owns: Chunks, Octrees (SDF), dirty tracking, ChunkManager state machine
//   RENDERER owns: GPU resources, RenderProxy creation/destruction, indirect draw
//
// The World knows nothing about Vulkan. It manages chunk lifecycle and provides
// dirty notifications. The renderer queries dirty chunks, generates meshes,
// uploads to GPU, and swaps RenderProxy pointers back into the Chunks.
//
// Thread safety:
//   - chunkMap_ is protected by a mutex
//   - ChunkManager provides its own fine-grained locking
//   - The dirty chunk queue is a deque protected by the chunk mutex
//   - RenderProxy pointers in Chunk are shared_ptr (ref-counted, atomically swappable)
class World {
public:
    // Unique identifier for a chunk (typically the octree node ID cast to uint64).
    using ChunkId = uint64_t;

    // Callback invoked when a chunk is marked dirty.
    // The renderer hooks into this to schedule rebuilds.
    using DirtyCallback = std::function<void(ChunkId id, uint32_t version, uint32_t layer)>;

    World();
    ~World();

    // No copying.
    World(const World&) = delete;
    World& operator=(const World&) = delete;

    // ── Chunk management ────────────────────────────────────────────────────

    // Get or create a chunk with the given ID.
    Chunk* getOrCreateChunk(ChunkId id, uint32_t layer = 0);

    // Get an existing chunk (returns nullptr if not found).
    Chunk* getChunk(ChunkId id);
    const Chunk* getChunk(ChunkId id) const;

    // Remove a chunk (e.g., on node deletion).
    void removeChunk(ChunkId id);

    // Remove all chunks (scene cleanup).
    void removeAllChunks();

    // Mark a chunk as dirty. Triggers the async rebuild pipeline.
    // Returns true if this is the first queue (new work to do).
    bool markChunkDirty(ChunkId id, uint32_t version, uint32_t layer);

    // ── Scene/octree access ─────────────────────────────────────────────────

    // The underlying scene with octrees.
    LocalScene& scene() { return *scene_; }
    const LocalScene& scene() const { return *scene_; }

    // Convenience access to octrees.
    Octree& opaqueOctree() { return scene_->getOpaqueOctree(); }
    const Octree& opaqueOctree() const { return scene_->getOpaqueOctree(); }
    Octree& transparentOctree() { return scene_->transparentOctree; }
    const Octree& transparentOctree() const { return scene_->transparentOctree; }

    // ── Chunk manager (state machine for the async rebuild pipeline) ────────

    ChunkManager& chunkManager() { return chunkManager_; }
    const ChunkManager& chunkManager() const { return chunkManager_; }

    // ── Dirty callback ──────────────────────────────────────────────────────

    // Set the callback that the World invokes when a chunk is marked dirty.
    // The renderer should register here to initiate the async rebuild pipeline.
    void setDirtyCallback(DirtyCallback callback) {
        dirtyCallback_ = std::move(callback);
    }

    // ── Debug / query ───────────────────────────────────────────────────────

    // Visit every chunk with a valid RenderProxy.
    template<typename F>
    void visitChunks(F&& visitor) const {
        std::lock_guard<std::mutex> lock(chunkMutex_);
        for (const auto& [id, chunk] : chunkMap_) {
            if (chunk && chunk->currentProxy && chunk->currentProxy->isValid()) {
                std::forward<F>(visitor)(*chunk);
            }
        }
    }

    size_t chunkCount() const {
        std::lock_guard<std::mutex> lock(chunkMutex_);
        return chunkMap_.size();
    }

    // ── Swap notification ───────────────────────────────────────────────────
    // Called by the renderer after atomically swapping a new RenderProxy in.
    // Updates the Chunk's currentProxy pointer.
    void notifyProxySwapped(ChunkId id, std::shared_ptr<const RenderProxy> newProxy);

private:
    // The scene with octrees (SDF storage + meshing).
    std::unique_ptr<LocalScene> scene_;

    // Per-chunk data (world-owned, no GPU references).
    mutable std::mutex chunkMutex_;
    std::unordered_map<ChunkId, std::unique_ptr<Chunk>> chunkMap_;

    // Chunk state machine driving the async rebuild pipeline.
    ChunkManager chunkManager_;

    // Callback for dirty notifications.
    DirtyCallback dirtyCallback_;
};

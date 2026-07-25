#pragma once

#include "ChunkState.hpp"
#include "RenderProxy.hpp"
#include "../../space/ThreadPool.hpp"
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <deque>
#include <mutex>
#include <atomic>
#include <memory>

// Tracks per-chunk state and drives the async rebuild pipeline:
//
//   Dirty → Queued → BuildingCPU → UploadingGPU → ReadyToSwap → Swap
//
// The main thread never waits for mesh generation. Worker threads produce
// Geometry on the CPU and enqueue uploads to the UploadManager. When the
// GPU transfer completes, the proxy is atomically swapped into the live
// scene and the old proxy is deferred for destruction.
//
// Thread safety:
//   - stateMap_ is protected by a mutex for non-atomic operations
//   - dirtyQueue_ is a thread-safe MPSC queue
//   - swapQueue_ is consumed on the main thread only
//   - RenderProxy pointers are swapped atomically

class ChunkManager {
public:
    // Unique identifier for a chunk (typically the NodeID cast to uint64).
    using ChunkId = uint64_t;

    // Per-chunk tracking data.
    struct ChunkEntry {
        ChunkState state = ChunkState::Clean;
        std::shared_ptr<const RenderProxy> currentProxy; // currently rendering (immutable)
        std::shared_ptr<const RenderProxy> pendingProxy; // being built (nullptr until ReadyToSwap)
        uint32_t version = 0;       // octree version at last rebuild
        uint32_t rebuildCount = 0;  // monotonically increasing
        bool queuedForRebuild = false; // in dirty queue (prevents duplicates)
    };

    ChunkManager() = default;
    ~ChunkManager() = default;

    // ChunkManager is move-only.
    ChunkManager(const ChunkManager&) = delete;
    ChunkManager& operator=(const ChunkManager&) = delete;
    ChunkManager(ChunkManager&&) = default;
    ChunkManager& operator=(ChunkManager&&) = default;

    // ── Main thread API ─────────────────────────────────────────────────────

    // Mark a chunk as dirty. If it's already queued or building, the
    // request is coalesced (no duplicate entries). Returns true if this
    // is the first queue for this chunk (new work to do).
    bool markDirty(ChunkId id, uint32_t version);

    // Drain the swap queue: for each proxy that is ready, atomically
    // replace the current proxy and schedule the old one for deferred
    // destruction. Returns a list of swapped-out proxies whose GPU
    // resources must be destroyed after the current frame completes.
    // Call once per frame from the main thread.
    std::vector<std::shared_ptr<const RenderProxy>> processSwapQueue();

    // Get the current proxy for a chunk (thread-safe, lock-free via shared_ptr).
    std::shared_ptr<const RenderProxy> getCurrentProxy(ChunkId id) const;

    // Check if a chunk is currently being processed.
    bool isChunkBusy(ChunkId id) const;

    // Remove a chunk entirely (e.g., on scene unload).
    void removeChunk(ChunkId id);

    // Remove all chunks (scene cleanup).
    void removeAll();

    // ── Worker thread API ────────────────────────────────────────────────────

    // Transition a chunk from Queued → BuildingCPU.
    // Returns the current ChunkEntry for the worker to use.
    ChunkEntry beginBuild(ChunkId id);

    // Transition a chunk from BuildingCPU → UploadingGPU.
    // Stores the newly built proxy as pendingProxy.
    void finishBuild(ChunkId id, std::shared_ptr<RenderProxy> proxy);

    // Transition a chunk from UploadingGPU → ReadyToSwap.
    // Called when the GPU transfer for this chunk completes.
    void finishUpload(ChunkId id);

    // Transition a chunk from any state → Clean (cancel rebuild).
    void cancelRebuild(ChunkId id);

    // ── Query ───────────────────────────────────────────────────────────────

    size_t pendingSwapCount() const { return swapQueue_.size(); }
    size_t dirtyQueueSize() const;
    size_t activeChunkCount() const;

    // Visit all current proxies (for rendering).
    template<typename F>
    void visitCurrentProxies(F&& visitor) const {
        std::lock_guard<std::mutex> lock(mapMutex_);
        for (const auto& [id, entry] : stateMap_) {
            if (entry.currentProxy && entry.currentProxy->isValid()) {
                std::forward<F>(visitor)(*entry.currentProxy);
            }
        }
    }

private:
    // Set chunk state with optional version update.
    void setState(ChunkId id, ChunkState newState);

    mutable std::mutex mapMutex_;

    // Per-chunk state (protected by mapMutex_)
    std::unordered_map<ChunkId, ChunkEntry> stateMap_;

    // Dirty queue: chunks waiting to be picked up by workers.
    // Protected by mapMutex_ (we access the queuedForRebuild flag under it).
    std::deque<ChunkId> dirtyQueue_;

    // Swap queue: proxies ready to be swapped in.
    // Produced by finishUpload (from async callback), consumed on main thread.
    // Protected by a separate mutex to avoid contention on the rendering path.
    mutable std::mutex swapMutex_;
    std::deque<ChunkId> swapQueue_;
};


// ── Inline implementations ─────────────────────────────────────────────────

inline bool ChunkManager::markDirty(ChunkId id, uint32_t version) {
    std::lock_guard<std::mutex> lock(mapMutex_);
    auto [it, inserted] = stateMap_.try_emplace(id);
    ChunkEntry& entry = it->second;

    if (entry.queuedForRebuild) {
        // Already queued — nothing to do (coalesce)
        return false;
    }

    if (entry.state != ChunkState::Clean) {
        // Currently being built — mark it dirty again for after this build
        entry.version = version;
        return false;
    }

    entry.state = ChunkState::Queued;
    entry.version = version;
    entry.queuedForRebuild = true;
    dirtyQueue_.push_back(id);
    return true;
}

inline std::shared_ptr<const RenderProxy> ChunkManager::getCurrentProxy(ChunkId id) const {
    std::lock_guard<std::mutex> lock(mapMutex_);
    auto it = stateMap_.find(id);
    if (it == stateMap_.end()) return nullptr;
    return it->second.currentProxy;
}

inline bool ChunkManager::isChunkBusy(ChunkId id) const {
    std::lock_guard<std::mutex> lock(mapMutex_);
    auto it = stateMap_.find(id);
    if (it == stateMap_.end()) return false;
    return it->second.state != ChunkState::Clean;
}

inline void ChunkManager::removeChunk(ChunkId id) {
    std::lock_guard<std::mutex> lock(mapMutex_);
    stateMap_.erase(id);
}

inline void ChunkManager::removeAll() {
    std::lock_guard<std::mutex> lock(mapMutex_);
    stateMap_.clear();
    dirtyQueue_.clear();
    std::lock_guard<std::mutex> slock(swapMutex_);
    swapQueue_.clear();
}

inline ChunkManager::ChunkEntry ChunkManager::beginBuild(ChunkId id) {
    std::lock_guard<std::mutex> lock(mapMutex_);
    auto it = stateMap_.find(id);
    if (it == stateMap_.end()) return ChunkEntry{};
    it->second.state = ChunkState::BuildingCPU;
    return it->second;
}

inline void ChunkManager::finishBuild(ChunkId id, std::shared_ptr<RenderProxy> proxy) {
    std::lock_guard<std::mutex> lock(mapMutex_);
    auto it = stateMap_.find(id);
    if (it == stateMap_.end()) return;
    it->second.state = ChunkState::UploadingGPU;
    it->second.pendingProxy = std::move(proxy);
}

inline void ChunkManager::finishUpload(ChunkId id) {
    {
        std::lock_guard<std::mutex> lock(mapMutex_);
        auto it = stateMap_.find(id);
        if (it == stateMap_.end()) return;
        it->second.state = ChunkState::ReadyToSwap;
    }
    // Push onto swap queue (separate mutex)
    std::lock_guard<std::mutex> lock(swapMutex_);
    swapQueue_.push_back(id);
}

inline void ChunkManager::cancelRebuild(ChunkId id) {
    std::lock_guard<std::mutex> lock(mapMutex_);
    auto it = stateMap_.find(id);
    if (it == stateMap_.end()) return;
    it->second.state = ChunkState::Clean;
    it->second.queuedForRebuild = false;
    it->second.pendingProxy.reset();
}

inline std::vector<std::shared_ptr<const RenderProxy>> ChunkManager::processSwapQueue() {
    // Collect swap queue entries
    std::deque<ChunkId> ready;
    {
        std::lock_guard<std::mutex> lock(swapMutex_);
        ready.swap(swapQueue_);
    }

    std::vector<std::shared_ptr<const RenderProxy>> retired;
    std::lock_guard<std::mutex> lock(mapMutex_);

    for (ChunkId id : ready) {
        auto it = stateMap_.find(id);
        if (it == stateMap_.end()) continue;

        ChunkEntry& entry = it->second;
        if (entry.state != ChunkState::ReadyToSwap) continue;
        if (!entry.pendingProxy) continue;

        // Atomically swap proxies: the old proxy becomes current,
        // the pending proxy becomes new current.
        if (entry.currentProxy) {
            // Retire the old proxy (will be destroyed after fence)
            retired.push_back(std::move(entry.currentProxy));
        }

        // Swap in the new proxy
        entry.currentProxy = std::move(entry.pendingProxy);
        entry.state = ChunkState::Clean;
        entry.queuedForRebuild = false;
        entry.rebuildCount++;

        // Check if chunk was dirtied again during the build
        if (entry.version > entry.currentProxy->version) {
            entry.state = ChunkState::Queued;
            entry.queuedForRebuild = true;
            dirtyQueue_.push_back(id);
        }
    }

    return retired;
}

inline size_t ChunkManager::dirtyQueueSize() const {
    std::lock_guard<std::mutex> lock(mapMutex_);
    return dirtyQueue_.size();
}

inline size_t ChunkManager::activeChunkCount() const {
    std::lock_guard<std::mutex> lock(mapMutex_);
    return stateMap_.size();
}

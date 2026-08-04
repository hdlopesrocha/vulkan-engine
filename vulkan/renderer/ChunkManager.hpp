#pragma once

#include "ChunkState.hpp"
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
// GPU transfer completes, the new mesh version is swapped into the live
// scene state (the slot data itself lives in the IndirectRenderer).
//
// Thread safety:
//   - stateMap_ is protected by a mutex for non-atomic operations
//   - dirtyQueue_ is a thread-safe MPSC queue
//   - swapQueue_ is consumed on the main thread only
//   - version bookkeeping is protected by mapMutex_

class ChunkManager {
public:
    // Unique identifier for a chunk (typically the NodeID cast to uint64).
    using ChunkId = uint64_t;

    // Per-chunk tracking data.
    struct ChunkEntry {
        ChunkState state = ChunkState::Clean;
        uint32_t currentVersion = 0; // octree version of the live mesh
        uint32_t pendingVersion = 0; // octree version being built (ReadyToSwap)
        uint32_t slotIndex = UINT32_MAX; // stable IndirectRenderer slot (assigned on upload)
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

    // Drain the swap queue: for each chunk that is ready, record the new
    // mesh version as current. GPU resources are owned by the
    // IndirectRenderer slots and are never held here.
    // Call once per frame from the main thread.
    void processSwapQueue();

    // Check if a chunk is currently being processed.
    bool isChunkBusy(ChunkId id) const;

    // Remove a chunk entirely (e.g., on scene unload).
    void removeChunk(ChunkId id);

    // Remove all chunks (scene cleanup).
    void removeAll();

    // Store the IndirectRenderer slot index for a chunk (called after
    // addMeshSlotted in processPendingMeshes). The slot index is needed
    // by the erase path to free the slot via removeMeshSlotted.
    void setSlotIndex(ChunkId id, uint32_t slotIndex);

    // Get the stored slot index (UINT32_MAX if not yet assigned).
    uint32_t getSlotIndex(ChunkId id) const;

    // ── Worker thread API ────────────────────────────────────────────────────

    // Transition a chunk from Queued → BuildingCPU.
    // Returns the current ChunkEntry for the worker to use.
    ChunkEntry beginBuild(ChunkId id);

    // Transition a chunk from BuildingCPU → UploadingGPU.
    // Records the octree version of the mesh being uploaded.
    void finishBuild(ChunkId id, uint32_t version);

    // Transition a chunk from UploadingGPU → ReadyToSwap.
    // Called when the GPU transfer for this chunk completes.
    void finishUpload(ChunkId id);

    // Transition a chunk from any state → Clean (cancel rebuild).
    void cancelRebuild(ChunkId id);

    // ── Query ───────────────────────────────────────────────────────────────

    size_t dirtyQueueSize() const;
    size_t activeChunkCount() const;

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

inline void ChunkManager::finishBuild(ChunkId id, uint32_t version) {
    std::lock_guard<std::mutex> lock(mapMutex_);
    auto it = stateMap_.find(id);
    if (it == stateMap_.end()) return;
    it->second.state = ChunkState::UploadingGPU;
    it->second.pendingVersion = version;
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
    it->second.pendingVersion = 0;
}

inline void ChunkManager::processSwapQueue() {
    // Collect swap queue entries
    std::deque<ChunkId> ready;
    {
        std::lock_guard<std::mutex> lock(swapMutex_);
        ready.swap(swapQueue_);
    }

    std::lock_guard<std::mutex> lock(mapMutex_);

    for (ChunkId id : ready) {
        auto it = stateMap_.find(id);
        if (it == stateMap_.end()) continue;

        ChunkEntry& entry = it->second;
        if (entry.state != ChunkState::ReadyToSwap) continue;

        // The new mesh version becomes current; the GPU-side slot data was
        // already installed by the upload completion callback.
        entry.currentVersion = entry.pendingVersion;
        entry.state = ChunkState::Clean;
        entry.queuedForRebuild = false;
        entry.rebuildCount++;

        // Check if chunk was dirtied again during the build
        if (entry.version > entry.currentVersion) {
            entry.state = ChunkState::Queued;
            entry.queuedForRebuild = true;
            dirtyQueue_.push_back(id);
        }
    }
}

inline size_t ChunkManager::dirtyQueueSize() const {
    std::lock_guard<std::mutex> lock(mapMutex_);
    return dirtyQueue_.size();
}

inline void ChunkManager::setSlotIndex(ChunkId id, uint32_t slotIndex) {
    std::lock_guard<std::mutex> lock(mapMutex_);
    auto it = stateMap_.find(id);
    if (it != stateMap_.end()) {
        it->second.slotIndex = slotIndex;
    }
}

inline uint32_t ChunkManager::getSlotIndex(ChunkId id) const {
    std::lock_guard<std::mutex> lock(mapMutex_);
    auto it = stateMap_.find(id);
    if (it == stateMap_.end()) return UINT32_MAX;
    return it->second.slotIndex;
}

inline size_t ChunkManager::activeChunkCount() const {
    std::lock_guard<std::mutex> lock(mapMutex_);
    return stateMap_.size();
}

// ── Composite chunk ids ─────────────────────────────────────────────────────
// Each (chunk, LoD level) pair gets its own ChunkManager entry: the base
// chunk id (NodeID) occupies the high bits, the 0..kMaxChunkLevels-1 level
// occupies the low nibble. Helpers to pack/unpack.

inline ChunkManager::ChunkId chunkIdForLevel(ChunkManager::ChunkId base, int level) {
    return (base << 4) | static_cast<ChunkManager::ChunkId>(level & 0xF);
}
inline int chunkLevelOf(ChunkManager::ChunkId cid) { return static_cast<int>(cid & 0xF); }
inline ChunkManager::ChunkId chunkBaseOf(ChunkManager::ChunkId cid) { return cid >> 4; }

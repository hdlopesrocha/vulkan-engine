#pragma once

// Opaque chunk IDs for the wire protocol.
//
// The WS API must never leak octree node pointers: pointer values reveal
// process internals (ASLR layout) and are unsafe as identities because the
// octree allocator reuses node memory. This registry hands out small
// monotonic IDs (1, 2, 3, ...) and maps them back to live nodes.
//
// Lifecycle (all under one mutex, thread-safe):
//   - getOrAssign(ptr): live octree change handlers and snapshot walks call
//     this when emitting a record. A node keeps its ID while alive.
//   - remove(ptr): delete handlers call this; returns the ID to publish (or
//     0 when the node was never announced). Erasing on delete is what makes
//     reuse safe: if the allocator recycles the memory for a new node, the
//     next getOrAssign mints a FRESH id, so clients can never confuse the new
//     occupant with the deleted chunk.
//   - lookup(id) / verify(id, ptr): on-demand mesh requests resolve through
//     these; a stale id simply yields "unknown" instead of a dangling pointer.
//
// Lock ordering: callers may hold octree tree locks while calling into the
// registry (record creation during apply/snapshot walks). The registry never
// takes any other lock, so tree -> registry is a consistent partial order.

#include <cstdint>
#include <mutex>
#include <unordered_map>

class OctreeNode;

class ChunkIdRegistry {
public:
    // 0 is reserved as "invalid / unknown" and is never assigned.
    static constexpr uint64_t kInvalid = 0;

    uint64_t getOrAssign(OctreeNode* node) {
        if (!node) return kInvalid;
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = nodeToId_.find(node);
        if (it != nodeToId_.end()) return it->second;
        const uint64_t id = nextId_++;
        nodeToId_[node] = id;
        idToNode_[id] = node;
        return id;
    }

    uint64_t remove(OctreeNode* node) {
        if (!node) return kInvalid;
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = nodeToId_.find(node);
        if (it == nodeToId_.end()) return kInvalid;
        const uint64_t id = it->second;
        nodeToId_.erase(it);
        idToNode_.erase(id);
        return id;
    }

    OctreeNode* lookup(uint64_t id) {
        if (id == kInvalid) return nullptr;
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = idToNode_.find(id);
        return it != idToNode_.end() ? it->second : nullptr;
    }

    // True only when id currently names exactly ptr (both directions agree).
    // Guards against allocator reuse: ptr deleted + recycled for a new node
    // no longer verifies under the old id (remove() erased it, and any
    // re-assign minted a different id).
    bool verify(uint64_t id, OctreeNode* ptr) {
        if (id == kInvalid || !ptr) return false;
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = idToNode_.find(id);
        if (it == idToNode_.end() || it->second != ptr) return false;
        auto jt = nodeToId_.find(ptr);
        return jt != nodeToId_.end() && jt->second == id;
    }

private:
    std::mutex mutex_;
    uint64_t nextId_ = 1;
    std::unordered_map<OctreeNode*, uint64_t> nodeToId_;
    std::unordered_map<uint64_t, OctreeNode*> idToNode_;
};

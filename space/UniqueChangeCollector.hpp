#pragma once
#include <unordered_map>
#include <mutex>
#include <utility>
#include "../space/OctreeNodeData.hpp"
#include "../space/OctreeNode.hpp"
#include "../space/Octree.hpp"

// Replacement for the removed UniqueOctreeChangeHandler: the same per-node
// event deduplication, exposed as Octree's two handler lambdas
// (Octree::UpdateHandler / Octree::DeleteHandler) instead of an
// OctreeChangeHandler subclass.
//
//   - updateHandler records a node as ADDED, keeping the newest version when
//     the same node was recorded before (Octree::apply can touch a chunk
//     several times before dispatch — e.g. one apply per brush SDF entry).
//   - deleteHandler records a node as DELETED, unconditionally overriding any
//     earlier add (a node erased after being added ends up as deleted).
//
// No user callback runs from the octree worker threads: apply() invokes the
// lambdas during the (threaded) traversal, but dispatch() replays the final
// per-node state on the caller's thread, exactly once per node.
class UniqueChangeCollector {
public:
    UniqueChangeCollector()
        : updateHandler([this](const OctreeNodeData& data) { onNodeAdded(data); }),
          deleteHandler([this](const OctreeNodeData& data) { onNodeDeleted(data); }) {}

    // The two lambdas handed to Scene::action / Scene::loadScene and
    // Octree::apply. Both are safe to call from any thread; they only mutate
    // the internal dedup map.
    Octree::OctreeNodeDataHandler updateHandler;
    Octree::OctreeNodeDataHandler deleteHandler;

    // Replay each node's final state (added or deleted) into the two
    // callbacks, once per node, in arbitrary (unordered) order. Empties the
    // pending set. Call on the thread that owns the callbacks (main thread).
    void dispatch(const Octree::OctreeNodeDataHandler& onAdded, const Octree::OctreeNodeDataHandler& onDeleted) {
        std::unordered_map<NodeID, std::pair<OctreeNodeData, bool>> localUpdates;
        {
            std::lock_guard<std::mutex> guard(mtx);
            localUpdates = std::move(updates);
            updates.clear();
        }
        for (const auto& e : localUpdates) {
            if (e.second.second) { if (onAdded) onAdded(e.second.first); }
            else                 { if (onDeleted) onDeleted(e.second.first); }
        }
    }

    void clear() {
        std::lock_guard<std::mutex> guard(mtx);
        updates.clear();
    }

private:
    void onNodeAdded(const OctreeNodeData& data) {
        NodeID id = reinterpret_cast<NodeID>(data.node);
        std::lock_guard<std::mutex> guard(mtx);
        OctreeNodeData& existingData = updates[id].first;
        if (existingData.node == nullptr ||
            existingData.node->version < data.node->version) {
            updates[id] = {OctreeNodeData{data}, true};
        }
    }
    void onNodeDeleted(const OctreeNodeData& data) {
        NodeID id = reinterpret_cast<NodeID>(data.node);
        std::lock_guard<std::mutex> guard(mtx);
        updates[id] = {OctreeNodeData{data}, false};
    }

    mutable std::mutex mtx;
    std::unordered_map<NodeID, std::pair<OctreeNodeData, bool>> updates;
};

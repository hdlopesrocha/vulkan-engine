#pragma once
#include <unordered_map>
#include <mutex>
#include <utility>
#include "OctreeNodeData.hpp"
#include "OctreeNode.hpp"
#include "Octree.hpp"

// Replacement for the removed UniqueOctreeChangeHandler: the same per-node
// event deduplication, exposed as Octree's two handler lambdas
// (Octree::UpdateHandler / Octree::DeleteHandler) instead of an
// OctreeChangeHandler subclass.
//
//   - updateHandler records a node as ADDED. Adds arrive in traversal order
//     for a given address, so the last one wins: it carries the cube/level of
//     the node that currently occupies that address.
//   - deleteHandler records a node as DELETED, unconditionally overriding any
//     earlier add (a node erased after being added ends up as deleted).
//
// Address reuse is why both rules matter. The octree allocator recycles freed
// nodes, so a pending add for address A can be followed by A being freed
// (delete) and reallocated for a different chunk (add). Comparing
// `existing.node->version` cannot distinguish those: both entries point at
// the same address, so the second add would see v == v and be dropped. The
// version comparison is therefore deliberately gone -- the stored entry is
// simply replaced.
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
        // Last add wins. Do NOT compare versions through the stored pointer:
        // the allocator reuses addresses, so existing.node and data.node can
        // be the same object and the comparison would always be v < v, which
        // silently dropped the new occupant's add (missing chunks).
        updates[id] = {OctreeNodeData{data}, true};
    }
    void onNodeDeleted(const OctreeNodeData& data) {
        NodeID id = reinterpret_cast<NodeID>(data.node);
        std::lock_guard<std::mutex> guard(mtx);
        updates[id] = {OctreeNodeData{data}, false};
    }

    mutable std::mutex mtx;
    std::unordered_map<NodeID, std::pair<OctreeNodeData, bool>> updates;
};

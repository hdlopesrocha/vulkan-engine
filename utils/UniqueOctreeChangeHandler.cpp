#include "UniqueOctreeChangeHandler.hpp"

UniqueOctreeChangeHandler::UniqueOctreeChangeHandler(const OctreeChangeHandler &handler)
:handler(handler) {}


void UniqueOctreeChangeHandler::onNodeAdded(const OctreeNodeData& data) const {
    NodeID id = reinterpret_cast<NodeID>(data.node);
    added[id] = OctreeNodeData{data};
}
void UniqueOctreeChangeHandler::onNodeDeleted(const OctreeNodeData& data) const {
    NodeID id = reinterpret_cast<NodeID>(data.node);
    deleted[id] = OctreeNodeData{data};
}
void UniqueOctreeChangeHandler::handleEvents() {
    for (const auto& e : deleted) {
        handler.onNodeDeleted(e.second);  
    }
    for (const auto& e : added) {
        handler.onNodeAdded(e.second);
    }
}
void UniqueOctreeChangeHandler::clear() {
    added.clear();
    deleted.clear();
}
const std::unordered_map<NodeID, OctreeNodeData>& UniqueOctreeChangeHandler::allAdded() const {
    return added;
}
const std::unordered_map<NodeID, OctreeNodeData>& UniqueOctreeChangeHandler::allDeleted() const {
    return deleted;
}

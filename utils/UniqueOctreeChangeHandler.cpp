#include "UniqueOctreeChangeHandler.hpp"

UniqueOctreeChangeHandler::UniqueOctreeChangeHandler(const OctreeChangeHandler &handler_)
:handler(handler_) {}


void UniqueOctreeChangeHandler::onNodeAdded(const OctreeNodeData& data) const {
    NodeID id = reinterpret_cast<NodeID>(data.node);
    std::lock_guard<std::mutex> guard(mtx);

    OctreeNodeData& existingData = updates[id].first;
    if(existingData.node == nullptr ||
        existingData.node->version < data.node->version) {
        updates[id] = {OctreeNodeData{data} , true};
    }
}
void UniqueOctreeChangeHandler::onNodeDeleted(const OctreeNodeData& data) const {
    NodeID id = reinterpret_cast<NodeID>(data.node);
    std::lock_guard<std::mutex> guard(mtx);
    updates[id] = {OctreeNodeData{data} , false };
}
void UniqueOctreeChangeHandler::handleEvents() {
    std::unordered_map<NodeID, std::pair<OctreeNodeData, bool>> localUpdates;
    {
        std::lock_guard<std::mutex> guard(mtx);
        localUpdates = std::move(updates);
        updates.clear();
    }
    for (const auto& e : localUpdates) {
        if(e.second.second) {
            handler.onNodeAdded(e.second.first);
        }
        else {
            handler.onNodeDeleted(e.second.first);
        }
    }
}
void UniqueOctreeChangeHandler::clear() {
    std::lock_guard<std::mutex> guard(mtx);
    updates.clear();
}


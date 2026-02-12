#include "UniqueOctreeChangeHandler.hpp"

UniqueOctreeChangeHandler::UniqueOctreeChangeHandler(const OctreeChangeHandler &handler)
:handler(handler) {}


void UniqueOctreeChangeHandler::onNodeAdded(const OctreeNodeData& data) const {
    NodeID id = reinterpret_cast<NodeID>(data.node);
    std::lock_guard<std::mutex> guard(mtx);
    updates[id] = {OctreeNodeData{data} , true};
}
void UniqueOctreeChangeHandler::onNodeDeleted(const OctreeNodeData& data) const {
    NodeID id = reinterpret_cast<NodeID>(data.node);
    std::lock_guard<std::mutex> guard(mtx);
    updates[id] = {OctreeNodeData{data} , false };
}
void UniqueOctreeChangeHandler::handleEvents() {
    std::lock_guard<std::mutex> guard(mtx);
    for (const auto& e : updates) {
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


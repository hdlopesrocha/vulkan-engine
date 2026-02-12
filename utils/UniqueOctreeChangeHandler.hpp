#pragma once
#include <unordered_map>
#include "../space/OctreeNodeData.hpp"
#include "../space/OctreeNode.hpp"
#include "../space/OctreeChangeHandler.hpp"
#include <functional>
#include <mutex>

class UniqueOctreeChangeHandler : public OctreeChangeHandler {
public:
    const OctreeChangeHandler &handler;
    UniqueOctreeChangeHandler(const OctreeChangeHandler &handler);

    const OctreeNodeData* getAdded(NodeID id) const;
    const OctreeNodeData* getDeleted(NodeID id) const;
    void onNodeAdded(const OctreeNodeData& data) const override;
    void onNodeDeleted(const OctreeNodeData& data) const override;
    void handleEvents();
    void clear();
    const std::unordered_map<NodeID, OctreeNodeData>& allAdded() const;
    const std::unordered_map<NodeID, OctreeNodeData>& allDeleted() const;
private:
    mutable std::unordered_map<NodeID, std::pair<OctreeNodeData,bool>> updates;
    mutable std::mutex mtx;
};

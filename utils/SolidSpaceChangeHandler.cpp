#include "SolidSpaceChangeHandler.hpp"

SolidSpaceChangeHandler::SolidSpaceChangeHandler(
    OctreeLayer<InstanceData> * solidInfo
) {
    this->solidInfo = solidInfo;
};

void SolidSpaceChangeHandler::create(OctreeNodeData& nodeData) {
    if (onNodeCreated) {
        onNodeCreated(nodeData);
    }
};

void SolidSpaceChangeHandler::update(OctreeNodeData& nodeData) {
    if (onNodeUpdated) {
        onNodeUpdated(nodeData);
    }
};

void SolidSpaceChangeHandler::erase(OctreeNodeData& nodeData) {
    if (onNodeErased) {
        onNodeErased(nodeData);
    }
};

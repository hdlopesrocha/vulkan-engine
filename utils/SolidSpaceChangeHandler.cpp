#include "SolidSpaceChangeHandler.hpp"

// Default empty callback used to initialize reference members
static NodeDataCallback emptyNodeDataCallback;

SolidSpaceChangeHandler::SolidSpaceChangeHandler(
    OctreeLayer<InstanceData> * solidInfo
) : solidInfo(solidInfo), onNodeUpdated(emptyNodeDataCallback), onNodeCreated(emptyNodeDataCallback), onNodeErased(emptyNodeDataCallback) {
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

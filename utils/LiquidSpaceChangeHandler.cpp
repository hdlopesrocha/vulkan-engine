#include "LiquidSpaceChangeHandler.hpp"

// Default empty callback used to initialize reference members
static NodeDataCallback emptyNodeDataCallback;

LiquidSpaceChangeHandler::LiquidSpaceChangeHandler(
    OctreeLayer<InstanceData> * liquidInfo
) : liquidInfo(liquidInfo), onNodeUpdated(emptyNodeDataCallback), onNodeErased(emptyNodeDataCallback), onNodeCreated(emptyNodeDataCallback) {
};

void LiquidSpaceChangeHandler::create(OctreeNodeData& nodeData) {
    if (onNodeCreated) {
        onNodeCreated(nodeData);
    }
};

void LiquidSpaceChangeHandler::update(OctreeNodeData& nodeData) {
    if (onNodeUpdated) {
        onNodeUpdated(nodeData);
    }
};

void LiquidSpaceChangeHandler::erase(OctreeNodeData& nodeData) {
    if (onNodeErased) {
        onNodeErased(nodeData);
    }
};

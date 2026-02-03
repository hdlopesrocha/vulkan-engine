#include "LiquidSpaceChangeHandler.hpp"

LiquidSpaceChangeHandler::LiquidSpaceChangeHandler(
    OctreeLayer<InstanceData> * liquidInfo
) {
    this->liquidInfo = liquidInfo;
};

void LiquidSpaceChangeHandler::create(OctreeNodeData& nodeData) {
    if (onNodeUpdated) {
        onNodeUpdated(nodeData);
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

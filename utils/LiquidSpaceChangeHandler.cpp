#include "LiquidSpaceChangeHandler.hpp"

LiquidSpaceChangeHandler::LiquidSpaceChangeHandler(
    OctreeLayer<InstanceData> * liquidInfo
) {
    this->liquidInfo = liquidInfo;
};

void LiquidSpaceChangeHandler::create(OctreeNode* node) {

};

void LiquidSpaceChangeHandler::update(OctreeNode* node) {
    if (onNodeUpdated) {
        onNodeUpdated(node);
    }
};

void LiquidSpaceChangeHandler::erase(OctreeNode* node) {
    if (node != NULL) {
        // liquidInfo->erase(node);
        if (onNodeErased) {
            onNodeErased(node);
        }
    }
};

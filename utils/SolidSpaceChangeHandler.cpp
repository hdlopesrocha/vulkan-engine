#include "SolidSpaceChangeHandler.hpp"

SolidSpaceChangeHandler::SolidSpaceChangeHandler(
    OctreeLayer<InstanceData> * solidInfo
) {
    this->solidInfo = solidInfo;
};

void SolidSpaceChangeHandler::create(OctreeNode* node) {
    
};



void SolidSpaceChangeHandler::update(OctreeNode* node) {
    //std::cout << "SolidSpaceChangeHandler::update " << node->id << std::endl;
};

void SolidSpaceChangeHandler::erase(OctreeNode* node) {
    if(node!= NULL) {
        solidInfo->erase(node);
    }
};

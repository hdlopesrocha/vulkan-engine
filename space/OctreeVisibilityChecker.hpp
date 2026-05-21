#pragma once
#include "IteratorHandler.hpp"
#include "Octree.hpp"
#include "OctreeNode.hpp"
#include "../math/Frustum.hpp"
#include <mutex>

class OctreeVisibilityChecker : public IteratorHandler{
    Frustum frustum;
    glm::vec3 viewDir;
public:
    glm::vec3 sortPosition;
    std::vector<OctreeNodeData> visibleNodes;
    std::mutex mutex;
    OctreeVisibilityChecker();
    void update(glm::mat4 m);
    bool iterate(const Octree &tree, OctreeNodeData &params) override;
    void getOrder(const Octree &tree, OctreeNodeData &params, uint8_t order[8]) override;
};

 

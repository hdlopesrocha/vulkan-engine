#pragma once
#include "Octree.hpp"
#include "OctreeNode.hpp"
#include "../math/Frustum.hpp"
#include <mutex>
#include <vector>

class OctreeVisibilityChecker {
    Frustum frustum;
    glm::vec3 viewDir;
public:
    glm::vec3 sortPosition;
    std::vector<OctreeNodeData> visibleNodes;
    std::mutex mutex;
    OctreeVisibilityChecker();
    void update(glm::mat4 m);
    bool iterate(const Octree &tree, OctreeNodeData &params);
    void getOrder(const Octree &tree, OctreeNodeData &params, uint8_t order[8]);
};

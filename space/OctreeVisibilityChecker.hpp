#ifndef SPACE_OCTREE_VISIBILITY_CHECKER_HPP
#define SPACE_OCTREE_VISIBILITY_CHECKER_HPP

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
    void before(const Octree &tree, OctreeNodeData &params) override;
    void after(const Octree &tree, OctreeNodeData &params) override;
    bool test(const Octree &tree, OctreeNodeData &params) override;
    void getOrder(const Octree &tree, OctreeNodeData &params, uint8_t order[8]) override;
};

#endif

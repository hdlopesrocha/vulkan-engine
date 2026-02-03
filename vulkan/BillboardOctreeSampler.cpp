#include "BillboardOctreeSampler.hpp"
#include "../space/IteratorHandler.hpp"
#include <unordered_set>

class GrassIteratorHandler : public IteratorHandler {
public:
    std::vector<glm::vec3> positions;
    bool test(const Octree &tree, OctreeNodeData &params) override {
        // Only process leaf nodes
        return params.node && params.node->isLeaf();
    }
    void before(const Octree &tree, OctreeNodeData &params) override {
        if (params.node && params.node->vertex.texIndex == 4) {
            positions.push_back(params.node->vertex.position);
        }
    }
    void after(const Octree &, OctreeNodeData &) override {}
    void getOrder(const Octree &, OctreeNodeData &, uint8_t order[8]) override {
        for (int i = 0; i < 8; ++i) order[i] = i;
    }
};

std::vector<glm::vec3> BillboardOctreeSampler::collectGrassPositions(Octree& octree) {
    GrassIteratorHandler handler;
    octree.iterate(handler);
    return handler.positions;
}

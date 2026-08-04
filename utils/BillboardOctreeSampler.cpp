#include "BillboardOctreeSampler.hpp"
#include <unordered_set>

std::vector<glm::vec3> BillboardOctreeSampler::collectGrassPositions(Octree& octree) {
    std::vector<glm::vec3> positions;
    octree.iterate(
        [&positions](const Octree &treeRef, OctreeNodeData &params) {
            // Only process leaf nodes
            bool result = params.node && !params.node->isLeaf();
            if (params.node && params.node->vertex.brushIndex == 4) {
                positions.push_back(params.node->vertex.position);
            }
            return result;
        },
        [](const Octree &, OctreeNodeData &, uint8_t order[8]) {
            for (int i = 0; i < 8; ++i) order[i] = i;
        }
    );
    return positions;
}

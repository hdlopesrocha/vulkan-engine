#pragma once
#include <vector>
#include <glm/glm.hpp>
#include "../space/Octree.hpp"
#include "../space/OctreeNode.hpp"
#include "../math/Vertex.hpp"

// Utility to sample billboard positions from an octree where brushIndex == 4 (grass)
class BillboardOctreeSampler {
public:
    // Collects positions of all leaf nodes in the octree where texIndex == 4 (grass)
    static std::vector<glm::vec3> collectGrassPositions(Octree& octree);
};

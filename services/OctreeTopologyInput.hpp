#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

#include "OctreeGpuNode.hpp"

// Everything the GPU topology pass needs for one chunk. `nodes` is the
// flattened neighborhood reachable by the neighbor descent (the anchor's
// ancestor path plus the nodes inside the search domain), `cells` are the
// indices of the nodes scanned at targetLod (the walkLadder frontier).
struct OctreeTopologyInput {
    std::vector<OctreeGpuNode> nodes;
    std::vector<uint32_t> cells;
    int targetLod = 0;
    uint32_t fromId = 0;   // index of the anchor (chunk) node
    int32_t fromBrush = -1;
    glm::vec3 rootMin{0.0f}; // world root cube, used for the contains() test
    float rootSize = 0.0f;
};

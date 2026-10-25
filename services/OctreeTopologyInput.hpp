#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

#include "OctreeGpuNode.hpp"

// Everything the GPU topology pass needs for one chunk. `nodes` is the
// flattened neighborhood reachable by the neighbor descent (the anchor's
// ancestor path plus the nodes inside the search domain), `cells` are the
// indices of the surface-straddling nodes scanned at targetLod (the compacted
// walkLadder frontier), and `activeVertexCount` is the number of surface nodes
// that received a vertex slot (the exact vertex-span size).
struct OctreeTopologyInput {
    std::vector<OctreeGpuNode> nodes;
    std::vector<uint32_t> cells;
    int targetLod = 0;
    uint32_t fromId = 0;   // descent-start anchor: the first path node whose
                           // stored lod <= targetLod, or the chunk node when
                           // no ancestor stops earlier (see findCellAt)
    int32_t fromBrush = -1;
    glm::vec3 rootMin{0.0f}; // world root cube, used for the contains() test
    float rootSize = 0.0f;
    uint32_t activeVertexCount = 0; // surface-node slots packed into flags bits 1+
    uint32_t scanCellCount = 0;     // all targetLod cells before compaction (key sizing)
};

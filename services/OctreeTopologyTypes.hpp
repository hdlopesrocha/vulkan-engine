#pragma once
#include <vector>
#include <cstdint>
#include <functional>
#include <glm/glm.hpp>

// Flattened octree node uploaded to the GPU triangle-topology pass
// (shaders/renderer/octree/OctreeTriangles.comp). Field-for-field mirror of
// the shader's storage buffers, see services/OctreeTopologyCompute.cpp for the
// binding assignment. The tree is flattened by LocalScene, which knows the
// octree and its allocator.
struct OctreeGpuNode {
    float sdf[8];        // stored corner samples (may carry INFINITY sentinels)
    glm::vec3 cubeMin;   // node cube min
    float cubeSize;      // node cube edge length
    int32_t lod;         // stored, +1-shifted ladder level (0 = unset)
    int32_t type;        // 0 = Empty, 1 = Surface, 2 = Solid (SpaceType)
    int32_t brushIndex;
    int32_t flags;       // bit0 = leaf (no children block)
    glm::vec3 hsv;
    int32_t children[8]; // indices into the node array, -1 = absent
};

struct OctreeTopologyVertex {
    glm::vec3 position;
    glm::vec3 normal;
    int32_t brushIndex;
    glm::vec3 hsv;
};

struct OctreeTopologyResult {
    // Three consecutive entries per triangle, in shader emission order.
    std::vector<OctreeTopologyVertex> vertices;
    uint32_t vertexCount = 0;
    uint32_t triangleCount = 0;
    // Set when the fixed output arena overflowed: the result is incomplete and
    // the caller must fall back to the CPU walker.
    bool truncated = false;
};

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

// App installs this when the GPU service is available; null in headless/server
// builds, where LocalScene falls back to the CPU walker.
extern std::function<OctreeTopologyResult(const OctreeTopologyInput&)> gOctreeTopologyCompute;

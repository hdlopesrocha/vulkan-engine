#pragma once

#include <cstdint>

#include <glm/glm.hpp>

// Flattened octree node uploaded to the GPU triangle-topology pass
// (shaders/computation/octree/OctreeTriangles.comp). Field-for-field mirror of
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
    int32_t flags;       // bit0 = leaf (no children block); bits1+ = vertexSlot + 1
                         // (0 = node has no vertex slot; assigned by LocalScene
                         // during surface-cell compaction)
    // Per-vertex mapping painted by the TexturePositionPainter (TextureType + Custom UV).
    int32_t textureType;
    glm::vec2 data;
    glm::vec3 hsv;
    int32_t children[8]; // indices into the node array, -1 = absent
};

#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

// One lava/flame emitter sampled from a brush-4 chunk surface. The chunk
// vertex stream is GPU-only, so the anchor is resolved from the node's SDF
// crossing on the CPU during the build (position + normal only); SdfRenderer
// derives the flame scale/seed from its lava config.
struct LavaAnchor {
    glm::vec3 position = glm::vec3(0.0f);
    glm::vec3 normal = glm::vec3(0.0f, 1.0f, 0.0f);
    float seed = 0.0f;
};

// One published chunk mesh as seen by the renderer. Originally nested in
// Octree; the GPU-resident build writes the actual Cartesian stream into the
// renderer's packed pools, so this record carries only the draw metadata and
// the host-side material identity.
struct LoDMesh {
    uint8_t lod = 0;              // 0-based LoD level of this mesh (0 = frontier chunks)
    unsigned int version = 0;     // snapshot of node->version at generation time
    float cellSize = 0;           // the chunk's own cell size
    glm::vec3 boundsMin = glm::vec3(0.0f); // emitting cell's world bounds (band center + meta)
    glm::vec3 boundsMax = glm::vec3(0.0f);
    glm::vec3 boundsBase = glm::vec3(0.0f); // emitting chunk's own min corner (informational; the GPU rung gate derives parent cells from the tree root lattice)
    // GPU-resident builds (services/OctreeTopologyCompute): the compute pass
    // already wrote the chunk's deduped Cartesian vertices and index stream
    // straight into the renderer's packed pools. The host never receives that
    // vertex stream — only these counts and the node's material identity for
    // CPU-side metadata (RT proxies).
    uint32_t gpuSlot = UINT32_MAX;   // packed-pool slot holding the mesh
    uint32_t gpuVertexCount = 0;     // published extent (highest referenced slot + 1)
    uint32_t gpuIndexCount = 0;      // index stream length (== draw index count)
    uint32_t gpuBaseVertex = 0;      // first vertex-pool element of the span
    uint32_t gpuFirstIndex = 0;      // first index-pool element of the span
    // Brush-4 (lava/fire) surface samples resolved from the chunk's node SDFs
    // during the GPU build. SdfRenderer turns these into flame anchors; the
    // renderer never sees the Cartesian vertex stream.
    std::vector<LavaAnchor> lavaAnchors;
    int32_t brushIndex = 0;          // node material (metadata only)
    int32_t textureType = 0;         // node mapping mode (metadata only)
};

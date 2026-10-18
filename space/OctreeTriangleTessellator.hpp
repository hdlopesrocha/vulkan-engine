#pragma once
#include <cstddef>
#include <functional>
#include "../math/Vertex.hpp"
#include "../math/BoundingCube.hpp"
#include "OctreeNodeTriangleHandler.hpp"

class Octree;
class OctreeNode;
class ThreadContext;

// CPU triangle topology for the octree ("the adaptive walker").
//
// The runtime path in the app is the GPU port of this walk in
// shaders/renderer/octree/OctreeTriangles.comp (dispatched by
// services/OctreeTopologyCompute). This translation unit is the reference and
// headless implementation and is still required by:
//   - bin/server (no Vulkan linkage) for the chunk meshes it serves to the web
//     viewer and to RemoteScene clients,
//   - LocalScene's fallback when the GPU pass is disabled, reports arena
//     overflow, or the device cannot run it,
//   - the VULKAN_TOPOLOGY_SELFCHECK A/B verification of the shader.
// Keep the two in sync when the walk changes; the self-check exists to prove
// they agree.
//
// Cube/level identity: cubes are rebuilt root-consistently by replaying the
// parent-index chain (never the cached/drifting per-node cubes), and `parentOf`
// links (seeded by the tree traversal in IteratorHandler) are the only source
// of the chain. Both properties are load-bearing for bit-exact output.
//
// Owns one walk over a tree: construct it with the octree and call
// iterateTriangles() for each subtree to tessellate, e.g.
//   OctreeTriangleTessellator(tree).iterateTriangles(node, cube, level,
//                                                    handler, context, lod);
// The walk holds the tree's shared lock for its duration and nests safely
// (see OctreeSharedLock). The walk's cell and cube-edge types live in
// space/EdgeCell.hpp and space/EdgeSpan.hpp.
class OctreeTriangleTessellator {
    const Octree &tree;
public:
    explicit OctreeTriangleTessellator(const Octree &octree) : tree(octree) {}

    // Walks the subtree at `from` (pass the chunk node and its root-consistent
    // cube) and emits triangle vertices to `func`.
    //   targetLod: +1-shifted STORED ladder level to tessellate. 0 = legacy
    //              full-walk mode, 1 = frontier, k = ancestor. The level-k mesh
    //              is the aggregate of the cells at lod k inside the anchor.
    //   vertexFn:  optional vertex override (e.g. GPU-generated vertices). When
    //              empty, vertices come from SDF::getPosition/getNormalFromPosition.
    void iterateTriangles(OctreeNode * from,
                          const BoundingCube &fromCube,
                          int fromLevel,
                          OctreeNodeTriangleHandler &func,
                          ThreadContext * context,
                          int targetLod = 0,
                          const std::function<Vertex(OctreeNode*, const BoundingCube&)>& vertexFn = nullptr);

private:
    // One walk over the tree: its fixed inputs, the per-call state (anchor,
    // caches) and the steps of the walk, defined in the .cpp. iterateTriangles()
    // builds one per call. The steps are members rather than lambdas, so a call
    // no longer constructs them and the recursive steps (collectBreaks,
    // walkLadder) need no std::function.
    struct Walk;
};

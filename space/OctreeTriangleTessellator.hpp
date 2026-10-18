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
// (see OctreeSharedLock).
class OctreeTriangleTessellator {
    const Octree &tree;
public:
    // One cell visited by the walk: its node, its root-consistent cube and the
    // +1-shifted STORED ladder level it sits at.
    struct EdgeCell {
        OctreeNode *node = NULL;
        BoundingCube cube;
        int level = 0;

        // A cell is "surface at the walk's resolution": a ladder cell exactly
        // at targetLod. With targetLod >= 1 the descent already stops at cells
        // with lod == targetLod, so requiring lod equality here keeps coarse
        // ladder levels (internal, non-simplified nodes) emitting their own
        // cells while neighbors one level finer/coarser stay out. lod and
        // targetLod are both in the +1-shifted STORED space.
        bool isSurface(int targetLod) const;
    };

    // A sub-span of one cube edge: the edge's axis, the two other axes (u, v),
    // where the edge sits on those axes (fixedU/fixedV) and how far it extends
    // along the axis (start..end). eps is the degeneracy tolerance used when
    // splitting at neighboring LOD boundaries.
    struct EdgeSpan {
        int axis = 0;
        int u = 1;
        int v = 2;
        float fixedU = 0.0f;
        float fixedV = 0.0f;
        float start = 0.0f;
        float end = 0.0f;
        float eps = 1e-6f;
    };

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

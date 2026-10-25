#pragma once
#include <functional>
#include "../math/geometry/BoundingCube.hpp"
#include "OctreeNodeTriangleHandler.hpp"

class Octree;
class OctreeNode;

// CPU triangle topology for the octree ("the adaptive walker").
//
// The runtime path in the app is the GPU port of this walk in
// shaders/computation/octree/OctreeTriangles.comp (dispatched by
// services/OctreeTopologyCompute). This translation unit is the offline
// reference implementation and is used ONLY by the Blender exporter
// (utils/BlenderExport.cpp), which supplies the vertexFn that solves a cell's
// surface point/normal from its SDF (SDF::getPosition / getNormalFromPosition).
// The walker itself never solves a Cartesian vertex: the engine runtime stores
// and combines SDF samples only, and Cartesian vertices are GPU-only.
//
// Keep the topology rules in sync with the compute shader when either changes.
//
// Cube/level identity: cubes are rebuilt root-consistently by replaying the
// parent-index chain from the world root (never cached/drifting per-node
// cubes), which is what keeps every lookup bit-identical to a root descent.
//
// Owns one walk over a tree: construct it with the octree and call
// iterateTriangles() for each subtree to tessellate, e.g.
//   OctreeTriangleTessellator(tree).iterateTriangles(node, cube, level,
//                                                    handler, lod, vertexFn);
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
    //   vertexFn:  REQUIRED surface-point/normal solver. The Blender exporter
    //              passes the SDF solver; the walker itself never computes a
    //              Cartesian vertex.
    void iterateTriangles(OctreeNode * from,
                          const BoundingCube &fromCube,
                          OctreeNodeTriangleHandler &func,
                          int targetLod,
                          const std::function<CellVertex(OctreeNode*, const BoundingCube&)>& vertexFn);

private:
    // One walk over the tree: its fixed inputs, the per-call cell cache and the
    // steps of the walk, defined in the .cpp. iterateTriangles() builds one per
    // call. The steps are members rather than lambdas, so a call no longer
    // constructs them and the recursive steps (collectBreaks, walkLadder) need
    // no std::function.
    struct Walk;
};

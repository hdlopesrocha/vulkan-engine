#pragma once
#include "../math/BoundingCube.hpp"

class OctreeNode;

// One cell visited by the octree triangle walk (OctreeTriangleTessellator):
// its node, its root-consistent cube and the +1-shifted STORED ladder level it
// sits at.
//
// Cube/level identity: cubes are rebuilt root-consistently by replaying the
// parent-index chain (never the cached/drifting per-node cubes), which is
// load-bearing for bit-exact output — see OctreeTriangleTessellator.
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

    // Ordering used to pick which of the four cells adjacent to an edge owns
    // the emitted segment: finest cube first, then position, then the node
    // pointer as a stable tie-breaker, so the choice is deterministic across
    // runs and platforms.
    static bool ownerLess(const EdgeCell &a, const EdgeCell &b);
};

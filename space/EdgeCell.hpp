#pragma once
#include "../math/geometry/BoundingCube.hpp"
#include "CellVertex.hpp"
#include <glm/glm.hpp>

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

    // A cell contributes its vertex to a segment when it is a surface node, at
    // ANY ladder level. The walk's own resolution picks which cells are SCANNED
    // (walkLadder only scans cells at targetLod), but the four quadrants around
    // an edge are resolved by findCellAt, which stops at cells of the walk's
    // level or COARSER. Requiring lod equality here threw those coarser
    // quadrants away, so a chunk and its coarser neighbour emitted different
    // polygons on their shared edge and the surface opened up along every LoD
    // band boundary (a sky gap where the two levels meet). Keeping them makes
    // the coarse cell bridge against the finer ones — the per-edge stitching
    // the span splitting exists for — and both sides of the seam now emit the
    // same polygon from the same cells.
    bool isSurface() const;

    // Ordering used to pick which of the four cells adjacent to an edge owns
    // the emitted segment: finest cube first, then position, then the node
    // pointer as a stable tie-breaker, so the choice is deterministic across
    // runs and platforms.
    static bool ownerLess(const EdgeCell &a, const EdgeCell &b);
};

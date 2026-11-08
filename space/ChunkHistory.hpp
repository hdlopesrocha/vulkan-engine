#pragma once

#include <cstdint>
#include <vector>
#include <glm/glm.hpp>
#include "../math/geometry/BoundingCube.hpp"

class Octree;
class OctreeNode;

// One chunk-level work unit for Octree::apply. The chunk-driven apply starts
// its iteration here (instead of at the world root): a chunk-band node, or a
// terminal node above the chunk band (a coarse leaf that owns a whole region).
//
// `parent` is an index into ChunkHistory::ancestors(), -1 when the anchor is
// the root itself.
struct ChunkAnchor {
    OctreeNode * node = nullptr;
    BoundingCube cube;
    uint32_t level = 0;
    int32_t parent = -1;
};

// One node above the chunk band on the path to at least one anchor. These are
// the internal cells a chunk-driven apply does not visit top-down, so it must
// refresh them bottom-up (deepest first) after the anchors were processed: the
// SDF corners of a parent are the diagonal corners of its eight children, and
// an unrefreshed parent would keep the previous apply's field, type and LoD.
struct ChunkAncestor {
    OctreeNode * node = nullptr;
    BoundingCube cube;
    uint32_t level = 0;
    // Index into ChunkHistory::ancestors(), -1 for the root.
    int32_t parent = -1;
};

// Per-thread history of the octree's chunk-level work units, stored in
// OctreeThreadContext and refreshed from the tree structure at the start of an
// apply. It is a history in the sense that the anchor list survives between
// applies (the vector keeps its capacity and the anchors are reused while the
// tree structure is unchanged); a chunk
// subtree the new shape cannot reach is not descended into. The anchor node
// itself still runs its corner combine, because the parent's child
// materialization writes that result back.
//
// refresh() is structural only: it never evaluates an SDF. It walks the tree
// above the chunk band and records every chunk-band node, plus every terminal
// node above it. The resulting lists are complete for the current tree, so an
// apply can start from the anchors and rebuild all ancestors exactly.
class ChunkHistory {
public:
    void refresh(const Octree &tree);
    void clear();

    const std::vector<ChunkAnchor> & anchors() const { return anchors_; }
    std::vector<ChunkAncestor> & ancestors() { return ancestors_; }
    const std::vector<ChunkAncestor> & ancestors() const { return ancestors_; }

private:
    void collect(const Octree &tree, OctreeNode *node, const BoundingCube &cube,
                 uint32_t level, int32_t parent);

    std::vector<ChunkAnchor> anchors_;
    std::vector<ChunkAncestor> ancestors_;
};

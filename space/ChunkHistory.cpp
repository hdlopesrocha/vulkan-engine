#include "ChunkHistory.hpp"
#include "Octree.hpp"
#include "OctreeNode.hpp"

void ChunkHistory::clear() {
    anchors_.clear();
    ancestors_.clear();
}

void ChunkHistory::refresh(const Octree &tree) {
    clear();
    if (tree.root == nullptr) {
        return;
    }
    collect(tree, tree.root, static_cast<const BoundingCube &>(tree), 0u, -1);
}

void ChunkHistory::collect(const Octree &tree, OctreeNode *node, const BoundingCube &cube,
                           uint32_t level, int32_t parent) {
    if (node == nullptr) {
        return;
    }

    // Chunk-band nodes are the natural work units: their subtree is what a
    // chunk-driven apply processes or reuses. A terminal node above the band is
    // a coarse work unit of its own (an empty/solid island or an unprocessed
    // root), and a node whose child block exists but is empty is treated the
    // same way so the anchor list can never silently drop a region.
    if (tree.isChunkNode(cube.getLengthX()) || node->isLeaf()) {
        anchors_.push_back(ChunkAnchor{node, cube, level, parent});
        return;
    }

    OctreeNode *children[8] = { nullptr, nullptr, nullptr, nullptr,
                                nullptr, nullptr, nullptr, nullptr };
    node->getChildren(*tree.allocator, children);
    bool hasChild = false;
    for (uint i = 0; i < 8; ++i) {
        hasChild |= children[i] != nullptr;
    }
    if (!hasChild) {
        anchors_.push_back(ChunkAnchor{node, cube, level, parent});
        return;
    }

    // Internal node above the chunk band: part of every anchor's parent chain.
    const int32_t index = static_cast<int32_t>(ancestors_.size());
    ancestors_.push_back(ChunkAncestor{node, cube, level, parent});
    for (uint i = 0; i < 8; ++i) {
        if (children[i] != nullptr) {
            collect(tree, children[i], cube.getChild(static_cast<int>(i)), level + 1u, index);
        }
    }
}

#ifndef SPACE_OCTREE_ALLOCATOR_HPP
#define SPACE_OCTREE_ALLOCATOR_HPP

#include "Allocator.hpp"

class OctreeNode;
class ChildBlock;

class OctreeAllocator {
public:
    Allocator<OctreeNode> nodeAllocator;
    Allocator<ChildBlock> childAllocator;

    OctreeAllocator();

    OctreeNode * allocate();
    void get(OctreeNode * nodes[8], uint indices[8]);
    OctreeNode * get(uint index);
    OctreeNode * deallocate(OctreeNode * node);
    uint getIndex(OctreeNode * node);
    size_t getBlockSize() const;
    size_t getAllocatedBlocksCount();
};

#endif // SPACE_OCTREE_ALLOCATOR_HPP

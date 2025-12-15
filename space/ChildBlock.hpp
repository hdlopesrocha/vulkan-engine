#ifndef SPACE_CHILD_BLOCK_HPP
#define SPACE_CHILD_BLOCK_HPP

#include "types.hpp"

struct ChildBlock {
    uint children[8];
    ChildBlock();
    ChildBlock * init();
    void clear(OctreeAllocator &allocator, OctreeChangeHandler * handler);
    bool isEmpty();
    void set(uint i, OctreeNode * node, OctreeAllocator &allocator);
    OctreeNode * get(uint i, OctreeAllocator &allocator);
};

#endif

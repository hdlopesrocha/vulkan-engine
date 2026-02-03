#pragma once
#include "OctreeChangeHandler.hpp"

class OctreeAllocator;
class OctreeNode;

#include <cstdint>
#include <climits>

class OctreeChangeHandler;
class OctreeAllocator;
class OctreeNode;

typedef unsigned int uint;

const uint UINT_MAX_ARRAY [8] = {UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX,UINT_MAX};

struct ChildBlock {
    uint children[8];
    ChildBlock();
    ChildBlock * init();
    void clear(OctreeAllocator &allocator, const BoundingCube& cube);
    bool isEmpty();
    void set(uint i, OctreeNode * node, OctreeAllocator &allocator);
    OctreeNode * get(uint i, OctreeAllocator &allocator);
};

 

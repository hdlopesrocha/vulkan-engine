#pragma once

#include <cstdint>
#include <functional>
#include "../math/BoundingCube.hpp"
#include "../math/ContainmentType.hpp"

class OctreeNode;

struct OctreeNodeData {
public:
    uint level;
    OctreeNode * node;
    BoundingCube cube;
    void * context;
    OctreeNodeData(uint level, OctreeNode * node, const BoundingCube &cube, void * context);
    OctreeNodeData(const OctreeNodeData &data);
    OctreeNodeData();
};


// Auto-generated wrapper header for OctreeNodeData
#pragma once

#include <cstdint>
#include "../math/BoundingCube.hpp"
#include "../math/ContainmentType.hpp"

class OctreeNode;

struct OctreeNodeData {
public:
    unsigned int level;
    OctreeNode * node;
    BoundingCube cube;
    ContainmentType containmentType;
    void * context;
    float sdf[8];
    OctreeNodeData(unsigned int level, OctreeNode * node, BoundingCube cube, ContainmentType containmentType, void * context, float * sdf);
    OctreeNodeData(const OctreeNodeData &data);
    OctreeNodeData();
};

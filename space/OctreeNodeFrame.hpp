// Auto-generated wrapper header for OctreeNodeFrame
#pragma once

#include "OctreeNode.hpp"
#include "../math/BoundingCube.hpp"
#include "../math/BrushMode.hpp"

typedef unsigned int uint;

struct OctreeNodeFrame {
    OctreeNode* node;
    BoundingCube cube;
    uint level;
    float sdf[8];
    int brushIndex;
    bool interpolated;
    BoundingCube chunkCube;
    OctreeNodeFrame();
    OctreeNodeFrame(const OctreeNodeFrame &t);
    OctreeNodeFrame(OctreeNode* node, BoundingCube cube, uint level, float * sdf, int brushIndex, bool interpolated, BoundingCube chunkCube);
};


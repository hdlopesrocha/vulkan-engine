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
    BoundingCube chunkCube;
    OctreeNodeFrame();
    OctreeNodeFrame(const OctreeNodeFrame &t);
    OctreeNodeFrame(OctreeNode* node, BoundingCube cube, uint level, float * sdf, int brushIndex, BoundingCube chunkCube);
};


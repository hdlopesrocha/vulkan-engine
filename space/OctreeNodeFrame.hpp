#pragma once

#include "OctreeNode.hpp"
#include "../math/BoundingCube.hpp"
#include "../math/BrushMode.hpp"
#include <glm/glm.hpp>

typedef unsigned int uint;

struct OctreeNodeFrame {
    OctreeNode* node;
    OctreeNode * iteratedNode; // used for iteration to keep track of the current node being processed, can be different from node when iterating children
    BoundingCube cube;
    SpaceType type;
    uint level;
    float sdf[8];
    int brushIndex;
    glm::vec3 hsv;
    BoundingCube chunkCube;
    OctreeNodeFrame();
    OctreeNodeFrame(const OctreeNodeFrame &t);
    OctreeNodeFrame(OctreeNode* node, OctreeNode* iteratedNode, const BoundingCube &cube, SpaceType type, uint level, float * sdf, int brushIndex, const BoundingCube &chunkCube);
};


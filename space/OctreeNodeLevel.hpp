#pragma once

#include <cstdint>

class OctreeNode;

class OctreeNodeLevel {
public:
    OctreeNode* node;
    uint level;
    OctreeNodeLevel() : node(NULL), level(0) {}
    OctreeNodeLevel(OctreeNode* n, uint lvl) : node(n), level(lvl) {}
};

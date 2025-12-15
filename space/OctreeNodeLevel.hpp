// Auto-generated wrapper header for OctreeNodeLevel
#pragma once

#include <cstdint>

class OctreeNode;

class OctreeNodeLevel {
public:
    OctreeNode* node;
    uint level;
    OctreeNodeLevel() : node(NULL), level(0) {}
    OctreeNodeLevel(OctreeNode* node, uint level) : node(node), level(level) {}
};

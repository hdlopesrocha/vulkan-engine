#pragma once

#include <cstdint>
#include "../math/BoundingCube.hpp"
#include "../math/ContainmentType.hpp"

class OctreeNode;

struct OctreeNodeData {
public:
    uint level;
    OctreeNode * node;
    BoundingCube cube;
    ContainmentType containmentType;
    void * context;
    OctreeNodeData(uint level, OctreeNode * node, BoundingCube cube, ContainmentType containmentType, void * context);
    OctreeNodeData(const OctreeNodeData &data);
    OctreeNodeData();
};

using NodeDataCallback = std::function<void(const OctreeNodeData&)>;

#pragma once

#include <cstdint>
#include <functional>
#include "../math/geometry/BoundingCube.hpp"
#include "../math/geometry/ContainmentType.hpp"

class OctreeNode;

struct OctreeNodeData {
public:
    uint level;
    OctreeNode * node;
    BoundingCube cube;
    OctreeNodeData(uint level, OctreeNode * node, const BoundingCube &cube);
    OctreeNodeData(const OctreeNodeData &data) = default;
    OctreeNodeData();
};


#pragma once

#include "../math/BoundingCube.hpp"
#include "OctreeNodeData.hpp"
#include <functional>

class OctreeNode;


class OctreeChangeHandler {
public:
    virtual void create(OctreeNodeData& data) = 0;
    virtual void update(OctreeNodeData& data) = 0;
    virtual void erase(OctreeNodeData& data) = 0;
};
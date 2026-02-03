#pragma once

#include "../math/BoundingCube.hpp"
#include "OctreeNodeData.hpp"
#include <functional>

class OctreeNode;


class OctreeChangeHandler {
public:
    virtual void onNodeAdded(const OctreeNodeData& data) const = 0;
    virtual void onNodeDeleted(const OctreeNodeData& data) const = 0;
};
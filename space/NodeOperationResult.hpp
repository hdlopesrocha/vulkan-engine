// Auto-generated wrapper header for NodeOperationResult
#pragma once

#include "../math/SpaceType.hpp"
#include "../math/BrushMode.hpp"

class OctreeNode;

struct NodeOperationResult {
    OctreeNode * node;
    SpaceType shapeType;
    SpaceType resultType;
    bool process;
    float resultSDF[8];
    float shapeSDF[8];
    bool isSimplified;
    int brushIndex;
    NodeOperationResult();
    NodeOperationResult(OctreeNode * node, SpaceType shapeType, SpaceType resultType, const float * resultSDF, const float * shapeSDF, bool process, bool isSimplified, int brushIndex);
};
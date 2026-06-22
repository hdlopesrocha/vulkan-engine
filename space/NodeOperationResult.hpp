#pragma once

#include "../math/SpaceType.hpp"
#include "../math/BrushMode.hpp"
#include "../math/ContainmentType.hpp"
#include "../math/Math.hpp"
class OctreeNode;

struct NodeOperationResult {
    OctreeNode * node;
    SpaceType shapeType;
    SpaceType resultType;
    float resultSDF[8] = {INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY};
    float shapeSDF[8] = {INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY};
    bool isSimplified;
    bool isLeaf;
    bool isChunk;
    int brushIndex;
    ContainmentType check;
    NodeOperationResult();
    NodeOperationResult(
        OctreeNode * node, 
        SpaceType shapeType, 
        const float * shapeSDF, 
        SpaceType resultType, 
        const float * resultSDF, 
        bool isSimplified, 
        bool isLeaf,
        bool isChunk,
        int brushIndex);
};
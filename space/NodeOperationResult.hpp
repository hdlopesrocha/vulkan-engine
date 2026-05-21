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
    bool isLeaf;
    bool isChunk;
    int brushIndex;
    NodeOperationResult();
    NodeOperationResult(
        OctreeNode * node, 
        SpaceType shapeType, 
        const float * shapeSDF, 
        SpaceType resultType, 
        const float * resultSDF, 
        bool process, 
        bool isSimplified, 
        bool isLeaf,
        bool isChunk,
        int brushIndex);
};
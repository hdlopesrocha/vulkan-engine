#pragma once

#include "../math/SpaceType.hpp"
#include "../math/BrushMode.hpp"
#include "../math/ContainmentType.hpp"
#include "../math/Math.hpp"
#include "../sdf/SDF.hpp"
#include <glm/glm.hpp>
class OctreeNode;

struct NodeOperationResult {
    OctreeNode * node;
    SpaceType shapeType;
    SpaceType resultType;
    float resultSDF[8] = {SDF_FAR,SDF_FAR,SDF_FAR,SDF_FAR,SDF_FAR,SDF_FAR,SDF_FAR,SDF_FAR};
    float shapeSDF[8] = {SDF_FAR,SDF_FAR,SDF_FAR,SDF_FAR,SDF_FAR,SDF_FAR,SDF_FAR,SDF_FAR};
    float shapeSdfCenter = SDF_FAR;
    uint8_t isSimplified;
    bool isLeaf;
    bool isChunk;
    int brushIndex;
    glm::vec3 hsv;
    ContainmentType check;
    NodeOperationResult();
    NodeOperationResult(
        OctreeNode * node, 
        SpaceType shapeType, 
        const float * shapeSDF, 
        SpaceType resultType, 
        const float * resultSDF, 
        uint8_t isSimplified, 
        bool isLeaf,
        bool isChunk,
        int brushIndex);
};
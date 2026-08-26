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
    float resultSDF[8] = {INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY};
    float shapeSDF[8] = {INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY,INFINITY};
    int brushIndex;
    glm::vec3 brushHsv;

    uint8_t selectedLod;
    uint8_t selectedChunkLod;
    ContainmentType check;
    NodeOperationResult();
    NodeOperationResult(
        OctreeNode * node, 
        SpaceType shapeType, 
        const float * shapeSDF, 
        SpaceType resultType, 
        const float * resultSDF, 
        int brushIndex,
        uint8_t selectedLod,
        uint8_t selectedChunkLod);
};
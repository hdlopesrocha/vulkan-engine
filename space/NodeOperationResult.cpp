#include "NodeOperationResult.hpp"
#include <cstring>
#include <cmath>
#include "../math/BrushMode.hpp"

NodeOperationResult::NodeOperationResult()
    : node(NULL), shapeType(SpaceType::Empty), 
    resultType(SpaceType::Empty),  isSimplified(false), 
    isLeaf(false), isChunk(false), brushIndex(DISCARD_BRUSH_INDEX),
    check(ContainmentType::Disjoint)
{
    for(int i=0;i<8;++i){ resultSDF[i]=INFINITY; shapeSDF[i]=INFINITY; }
}

NodeOperationResult::NodeOperationResult(OctreeNode * node, SpaceType shapeType, const float * shapeSDF, SpaceType resultType, const float * resultSDF, bool isSimplified, bool isLeaf, bool isChunk, int brushIndex, ContainmentType check )
    : node(node), shapeType(shapeType), resultType(resultType), 
    isSimplified(isSimplified), isLeaf(isLeaf), 
    isChunk(isChunk), brushIndex(brushIndex), check(check)
{
    if (resultSDF) std::memcpy(this->resultSDF, resultSDF, sizeof(this->resultSDF)); else for(int i=0;i<8;++i) this->resultSDF[i]=INFINITY;
    if (shapeSDF) std::memcpy(this->shapeSDF, shapeSDF, sizeof(this->shapeSDF)); else for(int i=0;i<8;++i) this->shapeSDF[i]=INFINITY;
}

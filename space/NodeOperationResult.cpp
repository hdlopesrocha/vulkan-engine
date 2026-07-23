#include "NodeOperationResult.hpp"
#include <cstring>
#include <cmath>
#include "../math/BrushMode.hpp"



NodeOperationResult::NodeOperationResult()
    : node(NULL), shapeType(SpaceType::Empty), 
    resultType(SpaceType::Empty),  isSimplified(0u), 
    isLeaf(false), isChunk(false), brushIndex(DISCARD_BRUSH_INDEX)
{

}

NodeOperationResult::NodeOperationResult(OctreeNode * node_, SpaceType shapeType_, const float * shapeSDF_, SpaceType resultType_, const float * resultSDF_, uint8_t isSimplified_, bool isLeaf_, bool isChunk_, int brushIndex_)
    : node(node_), shapeType(shapeType_), resultType(resultType_), 
    isSimplified(isSimplified_), isLeaf(isLeaf_), 
    isChunk(isChunk_), brushIndex(brushIndex_)
{
    if (resultSDF_) 
        std::memcpy(this->resultSDF, resultSDF_, sizeof(this->resultSDF)); else for(int i=0;i<8;++i) this->resultSDF[i]=INFINITY;
    if (shapeSDF_) 
        std::memcpy(this->shapeSDF, shapeSDF_, sizeof(this->shapeSDF)); else for(int i=0;i<8;++i) this->shapeSDF[i]=INFINITY;
}

#include "NodeOperationResult.hpp"
#include <cstring>
#include <cmath>
#include "../math/BrushMode.hpp"

NodeOperationResult::NodeOperationResult()
    : node(NULL), shapeType(SpaceType::Empty), resultType(SpaceType::Empty), process(false), isSimplified(false), brushIndex(DISCARD_BRUSH_INDEX)
{
    for(int i=0;i<8;++i){ resultSDF[i]=INFINITY; shapeSDF[i]=INFINITY; }
}

NodeOperationResult::NodeOperationResult(OctreeNode * node, SpaceType shapeType, SpaceType resultType, const float * resultSDF, const float * shapeSDF, bool process, bool isSimplified, int brushIndex)
    : node(node), shapeType(shapeType), resultType(resultType), process(process), isSimplified(isSimplified), brushIndex(brushIndex)
{
    if (resultSDF) std::memcpy(this->resultSDF, resultSDF, sizeof(this->resultSDF)); else for(int i=0;i<8;++i) this->resultSDF[i]=INFINITY;
    if (shapeSDF) std::memcpy(this->shapeSDF, shapeSDF, sizeof(this->shapeSDF)); else for(int i=0;i<8;++i) this->shapeSDF[i]=INFINITY;
}

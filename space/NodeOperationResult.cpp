#include "NodeOperationResult.hpp"
#include <cmath>
#include "../math/BrushMode.hpp"



NodeOperationResult::NodeOperationResult(): 
    node(NULL), 
    shapeType(SpaceType::Empty), 
    resultType(SpaceType::Empty),   
    brushIndex(DISCARD_BRUSH_INDEX),
    brushHsv(0.0f, 0.5f, 0.5f),
    selectedLod(0u), 
    selectedChunkLod(0u)
{

}

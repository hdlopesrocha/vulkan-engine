#include "ShapeArgs.hpp"
#include "OctreeNodeData.hpp"

ShapeArgs::ShapeArgs(const SignedDistanceOperation &operation_, const SignedDistanceFunction &function_, const TexturePainter &painter_, const Transformation &model_, const Simplifier &simplifier_, float minSize_)
    : operation(&operation_), function(function_), painter(painter_), model(model_), simplifier(simplifier_), minSize(minSize_)
{
}

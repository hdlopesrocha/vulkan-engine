#include "ShapeArgs.hpp"
#include "OctreeChangeHandler.hpp"

ShapeArgs::ShapeArgs(float (*operation_)(float, float), WrappedSignedDistanceFunction * function_, const TexturePainter &painter_, const Transformation model_, glm::vec4 translate_, glm::vec4 scale_, Simplifier &simplifier_, const OctreeChangeHandler &changeHandler_, float minSize_)
    : operation(operation_), function(function_), painter(painter_), model(model_), translate(translate_), scale(scale_), simplifier(simplifier_), changeHandler(changeHandler_), minSize(minSize_)
{
}

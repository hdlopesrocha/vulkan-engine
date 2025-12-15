#include "ShapeArgs.hpp"
#include "OctreeChangeHandler.hpp"

ShapeArgs::ShapeArgs(float (*operation)(float, float), WrappedSignedDistanceFunction * function, const TexturePainter &painter, const Transformation model, glm::vec4 translate, glm::vec4 scale, Simplifier &simplifier, OctreeChangeHandler * changeHandler, float minSize)
    : operation(operation), function(function), painter(painter), model(model), translate(translate), scale(scale), simplifier(simplifier), changeHandler(changeHandler), minSize(minSize)
{
}

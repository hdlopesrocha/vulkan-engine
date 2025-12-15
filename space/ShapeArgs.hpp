// Auto-generated wrapper header for ShapeArgs
#pragma once

#include <glm/glm.hpp>
#include "../math/Transformation.hpp"
#include "../math/TexturePainter.hpp"
#include "../sdf/WrappedSignedDistanceFunction.hpp"
#include "../math/TexturePainter.hpp"
#include "Simplifier.hpp"
#include "OctreeChangeHandler.hpp"

class OctreeChangeHandler;

struct ShapeArgs {
    float (*operation)(float, float);
    WrappedSignedDistanceFunction * function;
    const TexturePainter &painter;
    const Transformation model;
    glm::vec4 translate;
    glm::vec4 scale;
    Simplifier &simplifier;
    OctreeChangeHandler * changeHandler;
    float minSize;

    ShapeArgs(
        float (*operation)(float, float),
        WrappedSignedDistanceFunction * function,
        const TexturePainter &painter,
        const Transformation model,
        glm::vec4 translate,
        glm::vec4 scale,
        Simplifier &simplifier,
        OctreeChangeHandler * changeHandler,
        float minSize
    );
};
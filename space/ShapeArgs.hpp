#pragma once

#include <glm/glm.hpp>
#include "../math/Transformation.hpp"
#include "../math/TexturePainter.hpp"
#include "../sdf/SignedDistanceFunction.hpp"
#include "../sdf/SignedDistanceOperation.hpp"
#include "../math/TexturePainter.hpp"
#include "Simplifier.hpp"
#include "OctreeChangeHandler.hpp"

class OctreeChangeHandler;

struct ShapeArgs {
    const SignedDistanceOperation * operation;
    SignedDistanceFunction * function;
    const TexturePainter &painter;
    const Transformation model;
    Simplifier &simplifier;
    const OctreeChangeHandler &changeHandler; // reference (non-null)
    float minSize;

    ShapeArgs(
        const SignedDistanceOperation &operation,
        SignedDistanceFunction * function,
        const TexturePainter &painter,
        const Transformation model,
        Simplifier &simplifier,
        const OctreeChangeHandler &changeHandler,
        float minSize
    );
};

#pragma once

#include <glm/glm.hpp>
#include "../math/Transformation.hpp"
#include "../math/TexturePainter.hpp"
#include "../sdf/SignedDistanceFunction.hpp"
#include "../sdf/SignedDistanceOperation.hpp"
#include "Simplifier.hpp"
#include "OctreeNodeData.hpp"
#include <vector>
#include <mutex>
#include <memory>


struct ShapeArgs {
    const SignedDistanceOperation * operation;
    const SignedDistanceFunction &function;
    const TexturePainter &painter;
    const Transformation &model;
    const Simplifier &simplifier;
    float minSize;

    ShapeArgs(
        const SignedDistanceOperation &operation,
        const SignedDistanceFunction &function,
        const TexturePainter &painter,
        const Transformation &model,
        const Simplifier &simplifier,
        float minSize
    );
};

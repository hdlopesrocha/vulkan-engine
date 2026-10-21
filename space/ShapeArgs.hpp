#pragma once

#include <glm/glm.hpp>
#include "../math/geometry/Transformation.hpp"
#include "../math/TexturePositionPainter.hpp"
#include "../math/TextureCoordPainter.hpp"
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
    const TexturePositionPainter &painter;
    const TextureCoordPainter &coordPainter;
    const Transformation &model;
    const Simplifier &simplifier;
    float minSize;

    ShapeArgs(
        const SignedDistanceOperation &operation,
        const SignedDistanceFunction &function,
        const TexturePositionPainter &painter,
        const TextureCoordPainter &coordPainter,
        const Transformation &model,
        const Simplifier &simplifier,
        float minSize
    );
};

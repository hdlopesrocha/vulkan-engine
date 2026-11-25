#pragma once

#include <glm/glm.hpp>
#include "../math/geometry/Transformation.hpp"
#include "../math/BrushPainter.hpp"
#include "../math/TextureCoordPainter.hpp"
#include "../sdf/core/SignedDistanceFunction.hpp"
#include "../sdf/operations/SignedDistanceOperation.hpp"
#include "Simplifier.hpp"
#include "OctreeNodeData.hpp"
#include <cstdint>
#include <vector>
#include <mutex>
#include <memory>


struct ShapeArgs {
    const SignedDistanceOperation * operation;
    const SignedDistanceFunction &function;
    const BrushPainter &painter;
    const TextureCoordPainter &coordPainter;
    const Transformation &model;
    const Simplifier &simplifier;
    float minSize;

    // Shape-cache lattice, set by Octree::apply after expand(). Corners and
    // centers of this apply's leaf cubes sit at cacheOrigin + n*cacheSpacing;
    // queries off that lattice (stored finer nodes from an earlier apply)
    // take the cache's float-keyed fallback. cacheSpacing == 0 disables the
    // lattice fast path; cacheGeneration is the apply's tag for recycled
    // thread contexts (0 = no apply, see ShapeSdfCache).
    glm::vec3 cacheOrigin{0.0f};
    float cacheSpacing{0.0f};
    uint32_t cacheGeneration{0};

    ShapeArgs(
        const SignedDistanceOperation &operation,
        const SignedDistanceFunction &function,
        const BrushPainter &painter,
        const TextureCoordPainter &coordPainter,
        const Transformation &model,
        const Simplifier &simplifier,
        float minSize
    );
};

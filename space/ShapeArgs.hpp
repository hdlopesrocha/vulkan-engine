#pragma once

#include <glm/glm.hpp>
#include "../math/geometry/Transformation.hpp"
#include "../math/TexturePositionPainter.hpp"
#include "../math/TextureCoordPainter.hpp"
#include "../sdf/SignedDistanceFunction.hpp"
#include "../sdf/SignedDistanceOperation.hpp"
#include "Simplifier.hpp"
#include "OctreeNodeData.hpp"
#include <atomic>
#include <cstdint>
#include <vector>
#include <mutex>
#include <memory>


// Per-apply shape-cache counters. Filled only when the setup diagnostics are
// enabled (VULKAN_SETUP_DIAG); nullptr means "do not count", so the normal
// path pays a single predictable branch.
struct ShapeCacheStats {
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
};


struct ShapeArgs {
    const SignedDistanceOperation * operation;
    const SignedDistanceFunction &function;
    const TexturePositionPainter &painter;
    const TextureCoordPainter &coordPainter;
    const Transformation &model;
    const Simplifier &simplifier;
    float minSize;

    // Shape-cache lattice, set by Octree::apply after expand(). Every shape
    // query is a corner or center of a tree cube, i.e. cacheOrigin +
    // n * cacheSpacing. cacheSpacing == 0 disables the lattice fast path;
    // cacheGeneration is the apply's tag for recycled thread contexts
    // (0 = no apply, see ShapeSdfCache).
    glm::vec3 cacheOrigin{0.0f};
    float cacheSpacing{0.0f};
    uint32_t cacheGeneration{0};
    ShapeCacheStats *cacheStats{nullptr};

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

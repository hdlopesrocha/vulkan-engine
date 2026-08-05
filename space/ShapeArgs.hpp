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

    // Chunk change events fired during the shape traversal are deferred here
    // and dispatched by Octree::apply only after the traversal fully unwound
    // (leafs → root), so handlers see fresh per-node lod values: lod is
    // propagated bottom-up on shape return order, so the last lod written
    // before dispatch is the final one.
    struct DeferredChunkEvent {
        bool added; // true = onNodeAdded, false = onNodeDeleted
        OctreeNodeData data;
    };
    // Shared storage: shapeChildren captures ShapeArgs by value for pool
    // threads, so a plain vector would give every worker its own copy and
    // events would be lost. All copies share this heap vector; the mutex
    // guards push_back from concurrent workers.
    std::shared_ptr<std::mutex> deferredEventsMutex = std::make_shared<std::mutex>();
    ShapeArgs(
        const SignedDistanceOperation &operation,
        const SignedDistanceFunction &function,
        const TexturePainter &painter,
        const Transformation &model,
        const Simplifier &simplifier,
        float minSize
    );
};

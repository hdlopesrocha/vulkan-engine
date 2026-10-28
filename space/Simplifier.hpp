#pragma once
#include "../math/geometry/BoundingCube.hpp"
#include "SimplificationResult.hpp"
#include <glm/glm.hpp>
#include <utility>

struct NodeOperationResult;

class Simplifier {
    float toleranceMultiplier;
public:
    Simplifier(float toleranceMultiplier);
    // `chunkCube` is the bounding cube of the GPU-upload chunk that this node
    // belongs to.  Direct children of the chunk root are never simplified to
    // guarantee consistent detail at chunk boundaries.
    SimplificationResult simplify(const BoundingCube &cube, const float * sdf, NodeOperationResult * children, const BoundingCube& chunkCube) const;
};

 

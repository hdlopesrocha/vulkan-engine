#pragma once
#include "../math/BoundingCube.hpp"
#include <glm/glm.hpp>
#include <utility>

struct NodeOperationResult;

// Result of attempting to simplify eight child nodes into their parent.
// `isSimplified` indicates whether simplification succeeded.
// `brushIndex` is the brush/material index to assign when simplified.
struct SimplificationResult {
    uint8_t isSimplified;
    SimplificationResult(uint8_t simplified)
        : isSimplified(simplified) {
    }
};

class Simplifier {
    float angle;
    float distance;
    bool texturing;
public:
    Simplifier(float angle, float distance, bool texturing);
    // `chunkCube` is the bounding cube of the GPU-upload chunk that this node
    // belongs to.  Direct children of the chunk root are never simplified to
    // guarantee consistent detail at chunk boundaries.
    SimplificationResult simplify(const BoundingCube &cube, const float * sdf, NodeOperationResult * children, const BoundingCube& chunkCube) const;
};

 

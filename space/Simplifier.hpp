#pragma once
#include "../math/BoundingCube.hpp"
#include <utility>

struct NodeOperationResult;

// Result of attempting to simplify eight child nodes into their parent.
// `isSimplified` indicates whether simplification succeeded.
// `brushIndex` is the brush/material index to assign when simplified.
struct SimplificationResult {
    bool isSimplified;
    int brushIndex;
    SimplificationResult(bool isSimplified, int brushIndex) : isSimplified(isSimplified), brushIndex(brushIndex) {
        
    }
};

class Simplifier {
    float angle;
    float distance;
    bool texturing;
public:
    Simplifier(float angle, float distance, bool texturing);
    SimplificationResult simplify(const BoundingCube cube, const float * sdf, NodeOperationResult * children);
};

 

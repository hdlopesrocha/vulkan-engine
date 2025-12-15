#ifndef SPACE_SIMPLIFIER_HPP
#define SPACE_SIMPLIFIER_HPP

#include "../math/BoundingCube.hpp"
#include <utility>

struct NodeOperationResult;

class Simplifier {
    float angle;
    float distance;
    bool texturing;
public:
    Simplifier(float angle, float distance, bool texturing);
    std::pair<bool,int> simplify(const BoundingCube chunkCube, const BoundingCube cube, const float * sdf, NodeOperationResult * children);
};

#endif

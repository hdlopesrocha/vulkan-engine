// Auto-generated wrapper header for BoundingCubeHasher
#pragma once

#include "BoundingCube.hpp"


struct BoundingCubeHasher {
    std::size_t operator()(const BoundingCube &v) const {
        std::size_t hash = 0;
        hash ^= std::hash<glm::vec3>{}(v.getMin()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<float>{}(v.getLengthX()) + 0x01000193 + (hash << 6) + (hash >> 2);
        return hash;
    }
};

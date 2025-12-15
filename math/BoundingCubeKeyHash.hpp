// Auto-generated wrapper header for BoundingCubeKeyHash
#pragma once

#include "BoundingCube.hpp"

struct BoundingCubeKeyHash {
    size_t operator()(const BoundingCube& key) const {
        size_t h1 = std::hash<int>{}(key.getMinX());
        size_t h2 = std::hash<int>{}(key.getMinY());
        size_t h3 = std::hash<int>{}(key.getMinZ());
        size_t h4 = std::hash<int>{}(key.getLengthX());
        return h1 ^ h2 ^ h3 ^ h4;
    }
};
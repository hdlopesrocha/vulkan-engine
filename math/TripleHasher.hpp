#pragma once

#include "Triple.hpp"
#include <functional>

struct TripleHasher {
    std::size_t operator()(const Triple<uint,uint,uint> &v) const {
        std::size_t hash = 0;
        hash ^= std::hash<uint>{}(v.first) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<uint>{}(v.second) + 0x01000193 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<uint>{}(v.third) + 0x27d4eb2f + (hash << 6) + (hash >> 2);
        return hash;
    }
};
#ifndef MATH_COMMON_HPP
#define MATH_COMMON_HPP

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>
#include "Vertex.hpp"

const std::array<glm::ivec3, 8> CUBE_CORNERS = {{
    {0, 0, 0},
    {0, 0, 1},
    {0, 1, 0},
    {0, 1, 1},
    {1, 0, 0},
    {1, 0, 1},
    {1, 1, 0},
    {1, 1, 1}
}};

enum ContainmentType {
    Contains,
    Intersects,
    Disjoint
};

enum SpaceType {
    Empty,
    Surface,
    Solid
};

template<typename A, typename B, typename C>
struct Triple {
    A first;
    B second;
    C third;
    bool operator==(const Triple& other) const {
        return first == other.first && second == other.second && third == other.third;
    }
};

struct TripleHasher {
    std::size_t operator()(const Triple<uint,uint,uint> &v) const {
        std::size_t hash = 0;
        hash ^= std::hash<uint>{}(v.first) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<uint>{}(v.second) + 0x01000193 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<uint>{}(v.third) + 0x27d4eb2f + (hash << 6) + (hash >> 2);
        return hash;
    }
};

struct Triangle {
public:
    Vertex v[3];
    Triangle(Vertex v1, Vertex v2, Vertex v3){ v[0]=v1; v[1]=v2; v[2]=v3; }
    Triangle flip(){ Vertex t=v[1]; v[1]=v[2]; v[2]=t; return *this; }
};

#endif

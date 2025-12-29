#pragma once
#include "Vertex.hpp"
#include "VertexHasher.hpp"
#include <vector>
#include <type_traits>
#include <tsl/robin_map.h>

class Geometry
{

public:
    std::vector<Vertex> vertices;
    std::vector<uint> indices;
    tsl::robin_map<Vertex, size_t, VertexHasher> compactMap;

    // Calculates tangents for all vertices using indexed triangles
    void calculateTangents();

    glm::vec3 center;

    Geometry();
    ~Geometry();

    void addVertex(const Vertex &vertex);
    void addTriangle(const Vertex &v0, const Vertex &v1, const Vertex &v2);
    static glm::vec3 getNormal(Vertex * a, Vertex * b, Vertex * c);
    glm::vec3 getCenter();
    void setCenter();
};

 

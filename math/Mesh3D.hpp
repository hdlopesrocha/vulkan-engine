#pragma once

#include "Vertex.hpp"
#include <vector>
#include <glm/glm.hpp>

class Mesh3D {
public:
    Mesh3D() = default;

    // Set geometry
    void setGeometry(const std::vector<Vertex>& vertices, const std::vector<uint16_t>& indices);

    // Access geometry
    std::vector<Vertex>& getVertices() { return vertices; }
    const std::vector<uint16_t>& getIndices() const { return indices; }

    // Compute normals and tangents
    void computeNormals();
    void computeTangents();

private:
    std::vector<Vertex> vertices;
    std::vector<uint16_t> indices;
};

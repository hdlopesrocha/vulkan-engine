#pragma once

#include "../math/Vertex.hpp"
#include <vector>
#include <glm/glm.hpp>

class Model3D {
public:
    Model3D() = default;

    // Set geometry
    void setGeometry(const std::vector<Vertex>& vertices, const std::vector<uint16_t>& indices);

    // Access geometry
    const std::vector<Vertex>& getVertices() const { return vertices; }
    const std::vector<uint16_t>& getIndices() const { return indices; }

    // Compute normals and tangents
    void computeNormals();
    void computeTangents();

private:
    std::vector<Vertex> vertices;
    std::vector<uint16_t> indices;
};

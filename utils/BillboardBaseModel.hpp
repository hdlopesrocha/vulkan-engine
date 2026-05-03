#pragma once
#include <vector>
#include <glm/glm.hpp>
#include "../math/Vertex.hpp"

// 3D billboard model with 6 planes: 4 at 45° inclination + 2 vertical
// Allows vegetation to be visible from multiple angles without disappearing
class BillboardBaseModel {
public:
    BillboardBaseModel();
    ~BillboardBaseModel();

    // Returns vertices for the billboard base (positions, normals, texcoords)
    const std::vector<Vertex>& getVertices() const;
    // Returns indices for indexed drawing
    const std::vector<uint32_t>& getIndices() const;

private:
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    void createBase();
    void addQuad(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3);
};

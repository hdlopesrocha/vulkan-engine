#pragma once
#include <vector>
#include <vulkan/vulkan.h>
#include "../math/Vertex.hpp"

// Simple 3D model for a billboard base (flat quad or box)
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
};

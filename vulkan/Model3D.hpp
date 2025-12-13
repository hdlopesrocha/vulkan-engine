#pragma once

#include "vulkan.hpp"
#include <vector>
#include <glm/glm.hpp>

class Model3D {
public:
    Model3D() = default;

    // Build GPU buffers from provided vertices and indices
    // Automatically computes normals if not provided; per-fragment tangents are used instead of stored vertex tangents
    void build(VulkanApp* app, 
               const std::vector<Vertex>& vertices, 
               const std::vector<uint16_t>& indices,
               bool computeNormals = true,
               bool computeTangents = true);

    // Destroy GPU buffers
    void destroy(VkDevice device) { vbo.destroy(device); }

    const VertexBufferObject& getVBO() const { return vbo; }

protected:
    VertexBufferObject vbo{};

    // Helper methods for geometry computation
    static void computeNormals(std::vector<Vertex>& vertices, const std::vector<uint16_t>& indices);
};

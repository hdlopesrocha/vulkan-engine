#include "PlaneMesh.hpp"
#include <vector>
#include <glm/glm.hpp>

void PlaneMesh::build(VulkanApp* app, float width, float height, float texIndex) {
    // Create a simple quad in the XZ plane (Y=0) centered at origin
    float halfW = width * 0.5f;
    float halfH = height * 0.5f;
    
    // 4 vertices for a simple quad
    std::vector<Vertex> vertices = {
        // Bottom-left
        {{-halfW, 0.0f, -halfH}, {1,1,1}, {0.0f, 0.0f}, {0,1,0}, {1,0,0}, texIndex},
        // Bottom-right  
        {{ halfW, 0.0f, -halfH}, {1,1,1}, {1.0f, 0.0f}, {0,1,0}, {1,0,0}, texIndex},
        // Top-right
        {{ halfW, 0.0f,  halfH}, {1,1,1}, {1.0f, 1.0f}, {0,1,0}, {1,0,0}, texIndex},
        // Top-left
        {{-halfW, 0.0f,  halfH}, {1,1,1}, {0.0f, 1.0f}, {0,1,0}, {1,0,0}, texIndex},
    };

    // Two triangles to form the quad
    std::vector<uint16_t> indices = {
        0, 1, 2,  // First triangle
        2, 3, 0   // Second triangle
    };

    // Compute tangents for normal mapping
    // For a flat plane in XZ with normal (0,1,0), tangent is (1,0,0)
    for (auto& vertex : vertices) {
        vertex.tangent[0] = 1.0f;
        vertex.tangent[1] = 0.0f;
        vertex.tangent[2] = 0.0f;
    }

    Buffer vb = app->createVertexBuffer(vertices);
    Buffer ib = app->createIndexBuffer(indices);
    vbo = VertexBufferObject { vb, ib, static_cast<uint32_t>(indices.size()) };
}

void PlaneMesh::cleanup(VulkanApp* app) {
    if (vbo.vertexBuffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(app->getDevice(), vbo.vertexBuffer.buffer, nullptr);
        vkFreeMemory(app->getDevice(), vbo.vertexBuffer.memory, nullptr);
    }
    if (vbo.indexBuffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(app->getDevice(), vbo.indexBuffer.buffer, nullptr);
        vkFreeMemory(app->getDevice(), vbo.indexBuffer.memory, nullptr);
    }
}
#include "PlaneMesh.hpp"
#include <vector>

void PlaneMesh::build(VulkanApp* app, float width, float height, float texIndex) {
    // Create a simple quad in the XZ plane (Y=0) centered at origin
    float halfW = width * 0.5f;
    float halfH = height * 0.5f;
    
    // 4 vertices for a simple quad
    // UV coordinates scaled to repeat texture (5x5 repetition)
    float uvScale = 5.0f;
    std::vector<Vertex> vertices = {
        // Bottom-left
        {{-halfW, 0.0f, -halfH}, {1,1,1}, {0.0f, 0.0f}, {0,1,0}, {1,0,0,1.0f}, texIndex},
        // Bottom-right  
        {{ halfW, 0.0f, -halfH}, {1,1,1}, {uvScale, 0.0f}, {0,1,0}, {1,0,0,1.0f}, texIndex},
        // Top-right
        {{ halfW, 0.0f,  halfH}, {1,1,1}, {uvScale, uvScale}, {0,1,0}, {1,0,0,1.0f}, texIndex},
        // Top-left
        {{-halfW, 0.0f,  halfH}, {1,1,1}, {0.0f, uvScale}, {0,1,0}, {1,0,0,1.0f}, texIndex},
    };

    // Two triangles to form the quad
    std::vector<uint16_t> indices = {
        0, 1, 2,  // First triangle
        2, 3, 0   // Second triangle
    };

    // Use the base class build method and compute tangents on the CPU
    Model3D::build(app, vertices, indices, false, true);
}
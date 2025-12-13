#include "CubeMesh.hpp"
#include <vector>

void CubeMesh::build(VulkanApp* app, const std::vector<float>& faceTexIndices) {
    // 24 unique vertices (4 per face) so each face can have its own UVs
    // If faceTexIndices.size()==6, use those values per face; otherwise default to 0.0
    auto texForFace = [&](int face)->float{
        if (faceTexIndices.size() == 6) return faceTexIndices[face];
        return 0.0f;
    };
    
    std::vector<Vertex> vertices = {
        // +X face
        {{ 0.5f, -0.5f, -0.5f }, {1,1,1}, {0.0f,0.0f}, {1,0,0}, texForFace(0), {0,0,0,1}},
        {{ 0.5f, -0.5f,  0.5f }, {1,1,1}, {1.0f,0.0f}, {1,0,0}, texForFace(0), {0,0,0,1}},
        {{ 0.5f,  0.5f,  0.5f }, {1,1,1}, {1.0f,1.0f}, {1,0,0}, texForFace(0), {0,0,0,1}},
        {{ 0.5f,  0.5f, -0.5f }, {1,1,1}, {0.0f,1.0f}, {1,0,0}, texForFace(0), {0,0,0,1}},
        // -X face
        {{-0.5f, -0.5f,  0.5f }, {1,1,1}, {0.0f,0.0f}, {-1,0,0}, texForFace(1), {0,0,0,1}},
        {{-0.5f, -0.5f, -0.5f }, {1,1,1}, {1.0f,0.0f}, {-1,0,0}, texForFace(1), {0,0,0,1}},
        {{-0.5f,  0.5f, -0.5f }, {1,1,1}, {1.0f,1.0f}, {-1,0,0}, texForFace(1), {0,0,0,1}},
        {{-0.5f,  0.5f,  0.5f }, {1,1,1}, {0.0f,1.0f}, {-1,0,0}, texForFace(1), {0,0,0,1}},
        // +Y face
        {{-0.5f,  0.5f, -0.5f }, {1,1,1}, {0.0f,0.0f}, {0,1,0}, texForFace(2), {0,0,0,1}},
        {{ 0.5f,  0.5f, -0.5f }, {1,1,1}, {1.0f,0.0f}, {0,1,0}, texForFace(2), {0,0,0,1}},
        {{ 0.5f,  0.5f,  0.5f }, {1,1,1}, {1.0f,1.0f}, {0,1,0}, texForFace(2), {0,0,0,1}},
        {{-0.5f,  0.5f,  0.5f }, {1,1,1}, {0.0f,1.0f}, {0,1,0}, texForFace(2), {0,0,0,1}},
        // -Y face
        {{-0.5f, -0.5f,  0.5f }, {1,1,1}, {0.0f,0.0f}, {0,-1,0}, texForFace(3), {0,0,0,1}},
        {{ 0.5f, -0.5f,  0.5f }, {1,1,1}, {1.0f,0.0f}, {0,-1,0}, texForFace(3), {0,0,0,1}},
        {{ 0.5f, -0.5f, -0.5f }, {1,1,1}, {1.0f,1.0f}, {0,-1,0}, texForFace(3), {0,0,0,1}},
        {{-0.5f, -0.5f, -0.5f }, {1,1,1}, {0.0f,1.0f}, {0,-1,0}, texForFace(3), {0,0,0,1}},
        // +Z face
        {{-0.5f, -0.5f,  0.5f }, {1,1,1}, {0.0f,0.0f}, {0,0,1}, texForFace(4), {0,0,0,1}},
        {{-0.5f,  0.5f,  0.5f }, {1,1,1}, {1.0f,0.0f}, {0,0,1}, texForFace(4), {0,0,0,1}},
        {{ 0.5f,  0.5f,  0.5f }, {1,1,1}, {1.0f,1.0f}, {0,0,1}, texForFace(4), {0,0,0,1}},
        {{ 0.5f, -0.5f,  0.5f }, {1,1,1}, {0.0f,1.0f}, {0,0,1}, texForFace(4), {0,0,0,1}},
        // -Z face
        {{ 0.5f, -0.5f, -0.5f }, {1,1,1}, {0.0f,0.0f}, {0,0,-1}, texForFace(5), {0,0,0,1}},
        {{ 0.5f,  0.5f, -0.5f }, {1,1,1}, {1.0f,0.0f}, {0,0,-1}, texForFace(5), {0,0,0,1}},
        {{-0.5f,  0.5f, -0.5f }, {1,1,1}, {1.0f,1.0f}, {0,0,-1}, texForFace(5), {0,0,0,1}},
        {{-0.5f, -0.5f, -0.5f }, {1,1,1}, {0.0f,1.0f}, {0,0,-1}, texForFace(5), {0,0,0,1}},
    };

    std::vector<uint16_t> indices = {
        0,1,2, 2,3,0,         // +X
        4,5,6, 6,7,4,         // -X
        8,9,10, 10,11,8,      // +Y
        12,13,14, 14,15,12,   // -Y
        16,17,18, 18,19,16,   // +Z
        20,21,22, 22,23,20    // -Z
    };

    // Let the base class compute robust per-vertex tangents from positions/UVs
    Model3D::build(app, vertices, indices, false);
}

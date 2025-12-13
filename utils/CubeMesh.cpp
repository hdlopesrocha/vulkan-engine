#include "CubeMesh.hpp"
#include <vector>

void CubeMesh::build(const std::vector<float>& faceTexIndices) {
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

    // Set geometry and compute per-vertex tangents
    setGeometry(vertices, indices);
    //computeNormals();
    computeTangents();
}

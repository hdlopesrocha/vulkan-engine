#include "CubeMesh.hpp"
#include <vector>
#include <glm/glm.hpp>

void CubeMesh::build(VulkanApp* app, const std::vector<float>& faceTexIndices) {
    // 24 unique vertices (4 per face) so each face can have its own UVs
    // If faceTexIndices.size()==6, use those values per face; otherwise default to 0.0
    auto texForFace = [&](int face)->float{
        if (faceTexIndices.size() == 6) return faceTexIndices[face];
        return 0.0f;
    };
    std::vector<Vertex> vertices = {
        // +X face
        {{ 0.5f, -0.5f, -0.5f }, {1,1,1}, {0.0f,0.0f}, {1,0,0}, {0,0,0}, texForFace(0)},
        {{ 0.5f, -0.5f,  0.5f }, {1,1,1}, {1.0f,0.0f}, {1,0,0}, {0,0,0}, texForFace(0)},
        {{ 0.5f,  0.5f,  0.5f }, {1,1,1}, {1.0f,1.0f}, {1,0,0}, {0,0,0}, texForFace(0)},
        {{ 0.5f,  0.5f, -0.5f }, {1,1,1}, {0.0f,1.0f}, {1,0,0}, {0,0,0}, texForFace(0)},
        // -X face
        {{-0.5f, -0.5f,  0.5f }, {1,1,1}, {0.0f,0.0f}, {-1,0,0}, {0,0,0}, texForFace(1)},
        {{-0.5f, -0.5f, -0.5f }, {1,1,1}, {1.0f,0.0f}, {-1,0,0}, {0,0,0}, texForFace(1)},
        {{-0.5f,  0.5f, -0.5f }, {1,1,1}, {1.0f,1.0f}, {-1,0,0}, {0,0,0}, texForFace(1)},
        {{-0.5f,  0.5f,  0.5f }, {1,1,1}, {0.0f,1.0f}, {-1,0,0}, {0,0,0}, texForFace(1)},
        // +Y face
        {{-0.5f,  0.5f, -0.5f }, {1,1,1}, {0.0f,0.0f}, {0,1,0}, {0,0,0}, texForFace(2)},
        {{ 0.5f,  0.5f, -0.5f }, {1,1,1}, {1.0f,0.0f}, {0,1,0}, {0,0,0}, texForFace(2)},
        {{ 0.5f,  0.5f,  0.5f }, {1,1,1}, {1.0f,1.0f}, {0,1,0}, {0,0,0}, texForFace(2)},
        {{-0.5f,  0.5f,  0.5f }, {1,1,1}, {0.0f,1.0f}, {0,1,0}, {0,0,0}, texForFace(2)},
        // -Y face
        {{-0.5f, -0.5f,  0.5f }, {1,1,1}, {0.0f,0.0f}, {0,-1,0}, {0,0,0}, texForFace(3)},
        {{ 0.5f, -0.5f,  0.5f }, {1,1,1}, {1.0f,0.0f}, {0,-1,0}, {0,0,0}, texForFace(3)},
        {{ 0.5f, -0.5f, -0.5f }, {1,1,1}, {1.0f,1.0f}, {0,-1,0}, {0,0,0}, texForFace(3)},
        {{-0.5f, -0.5f, -0.5f }, {1,1,1}, {0.0f,1.0f}, {0,-1,0}, {0,0,0}, texForFace(3)},
        // +Z face
        {{-0.5f, -0.5f,  0.5f }, {1,1,1}, {0.0f,0.0f}, {0,0,1}, {0,0,0}, texForFace(4)},
        {{-0.5f,  0.5f,  0.5f }, {1,1,1}, {1.0f,0.0f}, {0,0,1}, {0,0,0}, texForFace(4)},
        {{ 0.5f,  0.5f,  0.5f }, {1,1,1}, {1.0f,1.0f}, {0,0,1}, {0,0,0}, texForFace(4)},
        {{ 0.5f, -0.5f,  0.5f }, {1,1,1}, {0.0f,1.0f}, {0,0,1}, {0,0,0}, texForFace(4)},
        // -Z face
        {{ 0.5f, -0.5f, -0.5f }, {1,1,1}, {0.0f,0.0f}, {0,0,-1}, {0,0,0}, texForFace(5)},
        {{ 0.5f,  0.5f, -0.5f }, {1,1,1}, {1.0f,0.0f}, {0,0,-1}, {0,0,0}, texForFace(5)},
        {{-0.5f,  0.5f, -0.5f }, {1,1,1}, {1.0f,1.0f}, {0,0,-1}, {0,0,0}, texForFace(5)},
        {{-0.5f, -0.5f, -0.5f }, {1,1,1}, {0.0f,1.0f}, {0,0,-1}, {0,0,0}, texForFace(5)},
    };

    std::vector<uint16_t> indices = {
        0,1,2, 2,3,0,         // +X
        4,5,6, 6,7,4,         // -X
        8,9,10, 10,11,8,      // +Y
        12,13,14, 14,15,12,   // -Y
        16,17,18, 18,19,16,   // +Z
        20,21,22, 22,23,20    // -Z
    };

    // compute smooth per-vertex normals by averaging face normals
    std::vector<glm::vec3> normAccum(vertices.size(), glm::vec3(0.0f));
    for (size_t i = 0; i < indices.size(); i += 3) {
        uint32_t i0 = indices[i];
        uint32_t i1 = indices[i+1];
        uint32_t i2 = indices[i+2];

        glm::vec3 p0(vertices[i0].pos[0], vertices[i0].pos[1], vertices[i0].pos[2]);
        glm::vec3 p1(vertices[i1].pos[0], vertices[i1].pos[1], vertices[i1].pos[2]);
        glm::vec3 p2(vertices[i2].pos[0], vertices[i2].pos[1], vertices[i2].pos[2]);

        glm::vec3 edge1 = p1 - p0;
        glm::vec3 edge2 = p2 - p0;
        glm::vec3 faceNormal = glm::cross(edge1, edge2);
        if (glm::length2(faceNormal) > 0.0f) faceNormal = glm::normalize(faceNormal);

        normAccum[i0] += faceNormal;
        normAccum[i1] += faceNormal;
        normAccum[i2] += faceNormal;
    }

    for (size_t i = 0; i < vertices.size(); ++i) {
        glm::vec3 n = normAccum[i];
        if (glm::length2(n) > 0.0f) n = glm::normalize(n);
        else n = glm::vec3(0.0f, 0.0f, 1.0f);
        vertices[i].normal[0] = n.x;
        vertices[i].normal[1] = n.y;
        vertices[i].normal[2] = n.z;
    }

    // compute per-vertex tangents from positions and UVs
    std::vector<glm::vec3> tanAccum(vertices.size(), glm::vec3(0.0f));
    for (size_t i = 0; i < indices.size(); i += 3) {
        uint32_t i0 = indices[i];
        uint32_t i1 = indices[i+1];
        uint32_t i2 = indices[i+2];

        glm::vec3 p0(vertices[i0].pos[0], vertices[i0].pos[1], vertices[i0].pos[2]);
        glm::vec3 p1(vertices[i1].pos[0], vertices[i1].pos[1], vertices[i1].pos[2]);
        glm::vec3 p2(vertices[i2].pos[0], vertices[i2].pos[1], vertices[i2].pos[2]);

        glm::vec2 uv0(vertices[i0].uv[0], vertices[i0].uv[1]);
        glm::vec2 uv1(vertices[i1].uv[0], vertices[i1].uv[1]);
        glm::vec2 uv2(vertices[i2].uv[0], vertices[i2].uv[1]);

        glm::vec3 edge1 = p1 - p0;
        glm::vec3 edge2 = p2 - p0;

        glm::vec2 deltaUV1 = uv1 - uv0;
        glm::vec2 deltaUV2 = uv2 - uv0;

        float denom = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;
        if (fabs(denom) < 1e-6f) continue;
        float f = 1.0f / denom;

        glm::vec3 tangent = f * (edge1 * deltaUV2.y - edge2 * deltaUV1.y);

        tanAccum[i0] += tangent;
        tanAccum[i1] += tangent;
        tanAccum[i2] += tangent;
    }

    // orthonormalize and store tangent per-vertex
    for (size_t i = 0; i < vertices.size(); ++i) {
        glm::vec3 n(vertices[i].normal[0], vertices[i].normal[1], vertices[i].normal[2]);
        glm::vec3 t = tanAccum[i];
        t = glm::normalize(t - n * glm::dot(n, t));
        if (!glm::isnan(t.x) && glm::length2(t) > 0.0f) {
            vertices[i].tangent[0] = t.x;
            vertices[i].tangent[1] = t.y;
            vertices[i].tangent[2] = t.z;
        } else {
            vertices[i].tangent[0] = 1.0f;
            vertices[i].tangent[1] = 0.0f;
            vertices[i].tangent[2] = 0.0f;
        }
    }

    Buffer vb = app->createVertexBuffer(vertices);
    Buffer ib = app->createIndexBuffer(indices);
    vbo = VertexBufferObject { vb, ib, (uint) indices.size() };
}

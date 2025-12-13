#include "Model3D.hpp"
#include <cmath>

void Model3D::build(VulkanApp* app, 
                    const std::vector<Vertex>& inputVertices, 
                    const std::vector<uint16_t>& inputIndices,
                    bool computeNormalsFlag) {
    
    // Make copies so we can modify them
    std::vector<Vertex> vertices = inputVertices;
    std::vector<uint16_t> indices = inputIndices;

    // Compute normals if requested
    if (computeNormalsFlag) {
        computeNormals(vertices, indices);
    }

    // Tangent computation removed; tangents are generated per-fragment if needed.

    // Create GPU buffers (createIndexBuffer expects non-const reference)
    Buffer vb = app->createVertexBuffer(vertices);
    Buffer ib = app->createIndexBuffer(indices);
    vbo = VertexBufferObject { vb, ib, static_cast<uint32_t>(indices.size()) };
}

void Model3D::computeNormals(std::vector<Vertex>& vertices, const std::vector<uint16_t>& indices) {
    // Compute smooth per-vertex normals by averaging face normals
    std::vector<glm::vec3> normAccum(vertices.size(), glm::vec3(0.0f));
    
    for (size_t i = 0; i < indices.size(); i += 3) {
        uint32_t i0 = indices[i];
        uint32_t i1 = indices[i+1];
        uint32_t i2 = indices[i+2];

        glm::vec3 p0(vertices[i0].pos[0], vertices[i0].pos[1], vertices[i0].pos[2]);
        glm::vec3 p1(vertices[i1].pos[0], vertices[i1].pos[1], vertices[i1].pos[2]);
        glm::vec3 p2(vertices[i2].pos[0], vertices[i2].pos[1], vertices[i2].pos[2]);

        glm::vec3 edge1 = p0 - p1;
        glm::vec3 edge2 = p0 - p2;
        glm::vec3 faceNormal = glm::cross(edge1, edge2);

        if (glm::length2(faceNormal) > 0.0f) {
            faceNormal = glm::normalize(faceNormal);
        }

        normAccum[i0] += faceNormal;
        normAccum[i1] += faceNormal;
        normAccum[i2] += faceNormal;
    }

    // Normalize and store accumulated normals
    for (size_t i = 0; i < vertices.size(); ++i) {
        glm::vec3 n = normAccum[i];
        if (glm::length2(n) > 0.0f) {
            n = glm::normalize(n);
        } else {
            n = glm::vec3(0.0f, 0.0f, 1.0f);
        }
        vertices[i].normal[0] = n.x;
        vertices[i].normal[1] = n.y;
        vertices[i].normal[2] = n.z;
    }
}


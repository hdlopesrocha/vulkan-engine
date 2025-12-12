#include "Model3D.hpp"
#include <cmath>

void Model3D::build(VulkanApp* app, 
                    const std::vector<Vertex>& inputVertices, 
                    const std::vector<uint16_t>& inputIndices,
                    bool computeNormalsFlag,
                    bool computeTangentsFlag) {
    
    // Make copies so we can modify them
    std::vector<Vertex> vertices = inputVertices;
    std::vector<uint16_t> indices = inputIndices;

    // Compute normals if requested
    if (computeNormalsFlag) {
        computeNormals(vertices, indices);
    }

    // Compute tangents if requested
    if (computeTangentsFlag) {
        computeTangents(vertices, indices);
    }

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

        glm::vec3 edge1 = p1 - p0;
        glm::vec3 edge2 = p2 - p0;
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

void Model3D::computeTangents(std::vector<Vertex>& vertices, const std::vector<uint16_t>& indices) {
    // Compute per-vertex tangents and bitangents from positions and UVs, and store handedness in tangent.w
    std::vector<glm::vec3> tanAccum(vertices.size(), glm::vec3(0.0f));
    std::vector<glm::vec3> bitAccum(vertices.size(), glm::vec3(0.0f));
    
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
        glm::vec3 bitangent = f * (edge2 * deltaUV1.x - edge1 * deltaUV2.x);

        tanAccum[i0] += tangent;
        tanAccum[i1] += tangent;
        tanAccum[i2] += tangent;

        bitAccum[i0] += bitangent;
        bitAccum[i1] += bitangent;
        bitAccum[i2] += bitangent;
    }

    // Orthonormalize and store tangent + handedness per-vertex
    for (size_t i = 0; i < vertices.size(); ++i) {
        glm::vec3 n(vertices[i].normal[0], vertices[i].normal[1], vertices[i].normal[2]);
        glm::vec3 t = tanAccum[i];
        glm::vec3 b = bitAccum[i];
        
        if (glm::length2(t) > 0.0f) {
            t = glm::normalize(t - n * glm::dot(n, t));
        }

        float handedness = 1.0f;
        if (glm::length2(b) > 0.0f && glm::length2(t) > 0.0f) {
            handedness = (glm::dot(glm::cross(n, t), b) < 0.0f) ? -1.0f : 1.0f;
        }

        if (!glm::isnan(t.x) && glm::length2(t) > 0.0f) {
            vertices[i].tangent[0] = t.x;
            vertices[i].tangent[1] = t.y;
            vertices[i].tangent[2] = t.z;
        } else {
            // Default tangent
            vertices[i].tangent[0] = 1.0f;
            vertices[i].tangent[1] = 0.0f;
            vertices[i].tangent[2] = 0.0f;
        }
    }
}

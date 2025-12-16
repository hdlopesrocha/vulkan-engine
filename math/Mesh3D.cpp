#include "Mesh3D.hpp"
#include <glm/gtc/constants.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/glm.hpp>
#include <cmath>

void Mesh3D::setGeometry(const std::vector<Vertex>& inVertices, const std::vector<uint16_t>& inIndices) {
    vertices = inVertices;
    indices = inIndices;
}

void Mesh3D::computeNormals() {
    std::vector<glm::vec3> normAccum(vertices.size(), glm::vec3(0.0f));
    for (size_t i = 0; i < indices.size(); i += 3) {
        uint32_t i0 = indices[i];
        uint32_t i1 = indices[i+1];
        uint32_t i2 = indices[i+2];

        glm::vec3 p0 = vertices[i0].position;
        glm::vec3 p1 = vertices[i1].position;
        glm::vec3 p2 = vertices[i2].position;

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

    for (size_t i = 0; i < vertices.size(); ++i) {
        glm::vec3 n = normAccum[i];
        if (glm::length2(n) > 0.0f) {
            n = glm::normalize(n);
        } else {
            n = glm::vec3(0.0f, 0.0f, 1.0f);
        }
        vertices[i].normal = n;
    }
}

void Mesh3D::computeTangents() {
    // compute per-vertex tangents using the same algorithm as before
    std::vector<glm::vec3> tanAccum(vertices.size(), glm::vec3(0.0f));
    std::vector<glm::vec3> bitAccum(vertices.size(), glm::vec3(0.0f));
    for (size_t i = 0; i < indices.size(); i += 3) {
        uint32_t i0 = indices[i];
        uint32_t i1 = indices[i+1];
        uint32_t i2 = indices[i+2];

        glm::vec3 p0 = vertices[i0].position;
        glm::vec3 p1 = vertices[i1].position;
        glm::vec3 p2 = vertices[i2].position;

        glm::vec2 uv0 = vertices[i0].texCoord;
        glm::vec2 uv1 = vertices[i1].texCoord;
        glm::vec2 uv2 = vertices[i2].texCoord;

        glm::vec3 edge1 = p1 - p0;
        glm::vec3 edge2 = p2 - p0;
        glm::vec2 duv1 = uv1 - uv0;
        glm::vec2 duv2 = uv2 - uv0;

        float denom = duv1.x * duv2.y - duv1.y * duv2.x;
        if (fabs(denom) < 1e-9f) continue;
        float r = 1.0f / denom;
        glm::vec3 tan = (edge1 * duv2.y - edge2 * duv1.y) * r;
        glm::vec3 bit = (edge2 * duv1.x - edge1 * duv2.x) * r;
        tanAccum[i0] += tan; tanAccum[i1] += tan; tanAccum[i2] += tan;
        bitAccum[i0] += bit; bitAccum[i1] += bit; bitAccum[i2] += bit;
    }

    for (size_t i = 0; i < vertices.size(); ++i) {
        glm::vec3 n = vertices[i].normal;
        glm::vec3 t = tanAccum[i];
        if (glm::length2(t) < 1e-8f) {
            glm::vec3 up = glm::abs(n.y) < 0.999f ? glm::vec3(0.0f,1.0f,0.0f) : glm::vec3(1.0f,0.0f,0.0f);
            t = glm::normalize(glm::cross(up, n));
        } else {
            t = glm::normalize(t - n * glm::dot(n, t));
        }
        glm::vec3 b = bitAccum[i];
        if (glm::length2(b) < 1e-8f) {
            b = glm::normalize(glm::cross(n, t));
        } else {
            b = glm::normalize(b - n * glm::dot(n, b));
        }
        float w = (glm::dot(glm::cross(n, t), b) < 0.0f) ? -1.0f : 1.0f;
        vertices[i].tangent = glm::vec4(t.x, t.y, t.z, w);
    }
}

#include "Geometry.hpp"
#include "Math.hpp"
#include "Vertex.hpp"
#include <glm/glm.hpp>
#include <cstddef>

void Geometry::calculateTangents() {
    if (vertices.empty() || indices.size() < 3) return;

    // --- Recompute per-vertex normals from face geometry (standard method) ---
    // Zero out vertex normals
    for (size_t i = 0; i < vertices.size(); ++i) {
        vertices[i].normal = glm::vec3(0.0f);
    }

    // Accumulate face normals (use consistent winding: v1-v0, v2-v0)
    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        size_t i0 = indices[i + 0];
        size_t i1 = indices[i + 1];
        size_t i2 = indices[i + 2];

        glm::vec3 p0 = vertices[i0].position;
        glm::vec3 p1 = vertices[i1].position;
        glm::vec3 p2 = vertices[i2].position;

        glm::vec3 face = glm::cross(p1 - p0, p2 - p0);
        float len2 = glm::dot(face, face);
        if (len2 > 1e-12f) {
            glm::vec3 fn = glm::normalize(face);
            vertices[i0].normal += fn;
            vertices[i1].normal += fn;
            vertices[i2].normal += fn;
        }
    }

    // Normalize accumulated vertex normals and fallback to (0,1,0) if degenerate
    for (size_t i = 0; i < vertices.size(); ++i) {
        float l2 = glm::dot(vertices[i].normal, vertices[i].normal);
        if (l2 > 1e-12f) vertices[i].normal = glm::normalize(vertices[i].normal);
        else vertices[i].normal = glm::vec3(0.0f, 1.0f, 0.0f);
    }

    // --- Tangent generation (unchanged, but uses new normals) ---
    std::vector<glm::vec3> tan1(vertices.size(), glm::vec3(0.0f));
    std::vector<glm::vec3> tan2(vertices.size(), glm::vec3(0.0f));

    // accumulate per-triangle contributions
    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        size_t i0 = indices[i + 0];
        size_t i1 = indices[i + 1];
        size_t i2 = indices[i + 2];

        const glm::vec3 &v0 = vertices[i0].position;
        const glm::vec3 &v1 = vertices[i1].position;
        const glm::vec3 &v2 = vertices[i2].position;

        const glm::vec2 &w0 = vertices[i0].texCoord;
        const glm::vec2 &w1 = vertices[i1].texCoord;
        const glm::vec2 &w2 = vertices[i2].texCoord;

        glm::vec3 edge1 = v1 - v0;
        glm::vec3 edge2 = v2 - v0;

        glm::vec2 deltaUV1 = w1 - w0;
        glm::vec2 deltaUV2 = w2 - w0;

        float denom = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;
        if (fabs(denom) <= 1e-8f) continue;
        float r = 1.0f / denom;

        glm::vec3 sdir = (edge1 * deltaUV2.y - edge2 * deltaUV1.y) * r;
        glm::vec3 tdir = (edge2 * deltaUV1.x - edge1 * deltaUV2.x) * r;

        tan1[i0] += sdir;
        tan1[i1] += sdir;
        tan1[i2] += sdir;

        tan2[i0] += tdir;
        tan2[i1] += tdir;
        tan2[i2] += tdir;
    }

    // Orthonormalize and store
    for (size_t i = 0; i < vertices.size(); ++i) {
        glm::vec3 n = vertices[i].normal;
        glm::vec3 t = tan1[i];

        // Gram-Schmidt orthogonalize
        if (glm::dot(t, t) > 1e-12f) {
            t = glm::normalize(t - n * glm::dot(n, t));
        } else {
            t = glm::vec3(1.0f, 0.0f, 0.0f); // fallback
        }

        // Calculate handedness
        glm::vec3 c = glm::cross(n, t);
        float w = (glm::dot(c, tan2[i]) < 0.0f) ? -1.0f : 1.0f;

        vertices[i].tangent = glm::vec4(t, w);
    }
}

Geometry::Geometry() {
    this->center = glm::vec3(0);
}

Geometry::~Geometry() {
}

glm::vec3 Geometry::getCenter(){
    return center;
}

void Geometry::setCenter(){
    if(vertices.size()){
        glm::vec3 min = vertices[0].position;
        glm::vec3 max = vertices[0].position;

        for(Vertex vertex : vertices) {
            glm::vec3 pos = vertex.position;
            min = glm::min(min, pos);
            max = glm::min(max, pos);
        }
        center = (min+max)*0.5f;
    }
    else {
        center = glm::vec3(0);
    }
}

void Geometry::addTriangle(const Vertex &v0, const Vertex &v1, const Vertex &v2) {
    // Preserve the winding order provided by the caller (v0, v1, v2).
    // The tessellator's `reverse` flag controls whether callers supply a reversed ordering.
    addVertex(v0);
    addVertex(v1);
    addVertex(v2);
}

void Geometry::addVertex(const Vertex &vertex) {
    auto [it, inserted] = compactMap.try_emplace(vertex, compactMap.size());
    size_t idx = it->second;
    
    if (inserted) {
        vertices.push_back(vertex);
    }
    indices.push_back(idx);
}

glm::vec3 Geometry::getNormal(Vertex * a, Vertex * b, Vertex * c) {
    glm::vec3 v1 = b->position-a->position;
    glm::vec3 v2 = c->position-a->position;
    return glm::normalize(glm::cross(v1 ,v2));
}

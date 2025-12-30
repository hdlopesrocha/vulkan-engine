#include "Geometry.hpp"
#include "Math.hpp"
#include "Vertex.hpp"
#include <glm/glm.hpp>
#include <cstddef>
#include <vector>

void Geometry::calculateTangents() {
    // Accumulate tangents and bitangents per vertex
    std::vector<glm::vec3> tan1(vertices.size(), glm::vec3(0.0f));
    std::vector<glm::vec3> tan2(vertices.size(), glm::vec3(0.0f));

    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        size_t i0 = indices[i];
        size_t i1 = indices[i+1];
        size_t i2 = indices[i+2];

        const glm::vec3& p0 = vertices[i0].position;
        const glm::vec3& p1 = vertices[i1].position;
        const glm::vec3& p2 = vertices[i2].position;

        const glm::vec2& uv0 = vertices[i0].texCoord;
        const glm::vec2& uv1 = vertices[i1].texCoord;
        const glm::vec2& uv2 = vertices[i2].texCoord;

        glm::vec3 edge1 = p1 - p0;
        glm::vec3 edge2 = p2 - p0;
        glm::vec2 deltaUV1 = uv1 - uv0;
        glm::vec2 deltaUV2 = uv2 - uv0;

        float f = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;
        float r = (f == 0.0f) ? 0.0f : 1.0f / f;

        glm::vec3 tangent = r * (edge1 * deltaUV2.y - edge2 * deltaUV1.y);
        glm::vec3 bitangent = r * (-edge1 * deltaUV2.x + edge2 * deltaUV1.x);

        tan1[i0] += tangent; tan1[i1] += tangent; tan1[i2] += tangent;
        tan2[i0] += bitangent; tan2[i1] += bitangent; tan2[i2] += bitangent;
    }

    // Orthonormalize and store per-vertex tangent + handedness
    for (size_t i = 0; i < vertices.size(); ++i) {
        glm::vec3 n = vertices[i].normal;
        glm::vec3 t = tan1[i];
        if (glm::length(t) > 0.0f) {
            // Gram-Schmidt orthogonalize
            t = glm::normalize(t - n * glm::dot(n, t));
            glm::vec3 b = tan2[i];
            float w = (glm::dot(glm::cross(n, t), b) < 0.0f) ? -1.0f : 1.0f;
            vertices[i].tangent = glm::vec4(t, w);
        } else {
            vertices[i].tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
        }
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
    addVertex(v0);
    addVertex(v2);
    addVertex(v1);
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

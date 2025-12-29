#include "Geometry.hpp"
#include "Math.hpp"
#include "Vertex.hpp"
#include <glm/glm.hpp>
#include <cstddef>

void Geometry::calculateTangents() {
    // Zero out all tangents
    for (auto& v : vertices) {
        v.tangent = glm::vec4(0.0f);
    }

    // Accumulate tangents per triangle
    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        Vertex& v0 = vertices[indices[i]];
        Vertex& v1 = vertices[indices[i+1]];
        Vertex& v2 = vertices[indices[i+2]];

        const glm::vec3& p0 = v0.position;
        const glm::vec3& p1 = v1.position;
        const glm::vec3& p2 = v2.position;

        const glm::vec2& uv0 = v0.texCoord;
        const glm::vec2& uv1 = v1.texCoord;
        const glm::vec2& uv2 = v2.texCoord;

        glm::vec3 edge1 = p1 - p0;
        glm::vec3 edge2 = p2 - p0;
        glm::vec2 deltaUV1 = uv1 - uv0;
        glm::vec2 deltaUV2 = uv2 - uv0;

        float f = deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y;
        float r = (f == 0.0f) ? 0.0f : 1.0f / f;

        glm::vec3 tangent = r * (edge1 * deltaUV2.y - edge2 * deltaUV1.y);

        v0.tangent += glm::vec4(tangent, 0.0f);
        v1.tangent += glm::vec4(tangent, 0.0f);
        v2.tangent += glm::vec4(tangent, 0.0f);
    }

    // Normalize tangents
    for (auto& v : vertices) {
        glm::vec3 t = glm::vec3(v.tangent);
        if (glm::length(t) > 0.0f)
            v.tangent = glm::vec4(glm::normalize(t), 0.0f);
        else
            v.tangent = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f); // fallback
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

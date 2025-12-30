#include "Geometry.hpp"
#include "Math.hpp"
#include "Vertex.hpp"
#include <glm/glm.hpp>
#include <cstddef>

void Geometry::calculateTangents() {
    // Tangent/bitangent are now computed in the fragment shader for triplanar
    // mapping; keep this function as a no-op to avoid CPU-side tangent storage.
    (void)vertices; (void)indices;
    return;
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

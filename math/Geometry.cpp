#include "Geometry.hpp"
#include "Math.hpp"
#include "Vertex.hpp"
#include <glm/glm.hpp>
#include <cstddef>

Geometry::Geometry() {
}

Geometry::Geometry(const Geometry& other)
    : vertices(other.vertices), indices(other.indices)
{
    // compactMap is dropped: it is only used transiently during build (addVertex)
    // and is not required once the geometry is complete.
}

Geometry& Geometry::operator=(const Geometry& other) {
    if (this != &other) {
        vertices = other.vertices;
        indices = other.indices;
        compactMap.reset();
    }
    return *this;
}

Geometry::~Geometry() {
}

void Geometry::addTriangle(const Vertex &v0, const Vertex &v1, const Vertex &v2) {
    // Preserve the winding order provided by the caller (v0, v1, v2).
    // The tessellator's `reverse` flag controls whether callers supply a reversed ordering.
    addVertex(v0);
    addVertex(v1);
    addVertex(v2);
}

void Geometry::addVertex(const Vertex &vertex) {
    if (!compactMap) {
        compactMap = std::make_unique<tsl::robin_map<Vertex, size_t, VertexHasher>>();
    }
    auto [it, inserted] = compactMap->try_emplace(vertex, compactMap->size());
    size_t idx = it->second;

    if (inserted) {
        vertices.push_back(vertex);
    }
    indices.push_back(idx);
}


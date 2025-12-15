#pragma once
#include "Vertex.hpp"
#include "VertexHasher.hpp"
#include <vector>
#include <type_traits>
#if __has_include(<tsl/robin_map.h>)
#include <tsl/robin_map.h>
#else
#include <unordered_map>
namespace tsl {
    template <typename K, typename V, typename H = std::hash<K>>
    using robin_map = std::unordered_map<K, V, H>;
}
#endif

class Geometry
{
public:
    std::vector<Vertex> vertices;
    std::vector<uint> indices;
    tsl::robin_map<Vertex, size_t, VertexHasher> compactMap;

    glm::vec3 center;
    bool reusable;

    Geometry(bool reusable);
    ~Geometry();

    void addVertex(const Vertex &vertex);
    void addTriangle(const Vertex &v0, const Vertex &v1, const Vertex &v2);
    static glm::vec3 getNormal(Vertex * a, Vertex * b, Vertex * c);
    glm::vec3 getCenter();
    void setCenter();
};

template <typename T> struct InstanceGeometry {
public:
    Geometry * geometry;
    std::vector<T> instances;
    InstanceGeometry(Geometry * geometry);
    InstanceGeometry(Geometry * geometry, std::vector<T> &instances);
    ~InstanceGeometry();
};

#include "Geometry.tpp"

 

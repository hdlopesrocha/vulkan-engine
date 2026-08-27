#pragma once
#include "Vertex.hpp"
#include "VertexHasher.hpp"
#include "BoundingCube.hpp"
#include <vector>
#include <type_traits>
#include <memory>
#include <tsl/robin_map.h>



class Geometry
{

public:
    std::vector<Vertex> vertices;
    std::vector<uint> indices;
    // Transient dedup map used only while the geometry is built via addVertex().
    // It is never needed after tessellation and must NOT be an inline
    // tsl::robin_map: its move/copy reads one byte past the enclosing object,
    // corrupting the heap metadata (crashes the app on free). Kept behind a
    // pointer so Geometry's move is a plain pointer swap.
    std::unique_ptr<tsl::robin_map<Vertex, size_t, VertexHasher>> compactMap;

    // Calculates tangents for all vertices using indexed triangles
    void calculateTangents();

    glm::vec3 center;

    Geometry();
    ~Geometry();
    Geometry(const Geometry& other);            // copies verts/indices/center; drops compactMap
    Geometry(Geometry&&) = default;
    Geometry& operator=(const Geometry& other); // same
    Geometry& operator=(Geometry&&) = default;

    void addVertex(const Vertex &vertex);
    void addTriangle(const Vertex &v0, const Vertex &v1, const Vertex &v2);
    static glm::vec3 getNormal(Vertex * a, Vertex * b, Vertex * c);
    glm::vec3 getCenter();
    void setCenter();
};

using GeometryLodCallback = std::function<void(
    const Geometry& geo,
    uint8_t lod,
    uint version,
    uintptr_t emittingNodeId,
    const BoundingCube& cube, // the emitting cell's OWN cube (band box, for frustum cull)
    const BoundingCube& baseCube // the FINEST chunk's cube (shared column anchor)
)>;
 

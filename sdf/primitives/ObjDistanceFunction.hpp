#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

#include "../core/SignedDistanceFunction.hpp"
#include "../../math/geometry/Transformation.hpp"
#include "../../math/geometry/BoundingSphere.hpp"

// Triangle soup loaded from a Wavefront OBJ file. Texture coordinates and
// normals are ignored: the SDF is a pure distance field and the brush paints
// with triplanar mapping. Faces are triangulated with a fan, and negative
// (relative) OBJ indices are supported.
struct ObjMesh {
    std::vector<glm::vec3> vertices;
    std::vector<std::array<uint32_t, 3>> triangles;
};

// Exact (up to float precision) Euclidean SDF of an arbitrary closed-ish
// triangle mesh, e.g. models/bunny.obj. The mesh is centred on the origin and
// uniformly scaled so its furthest vertex sits on the unit sphere, then baked
// into world space with the model transform — so non-uniform scale and
// rotation stay exact, matching the other mesh-derived primitives.
//
// The unsigned distance comes from a static BVH over the triangles. The sign
// uses angle-weighted pseudo-normals (Bærentzen & Aanæs): face normals in a
// triangle's interior, summed incident face normals on edges and angle-weighted
// vertex normals at vertices. This keeps the sign stable for meshes that are
// not perfectly closed, where a ray-parity test would flicker.
class ObjDistanceFunction : public SignedDistanceFunction {
public:
    // Loads `objPath` and builds the field. An unreadable/empty file leaves
    // the function empty (empty() == true) and the octree apply is a no-op.
    ObjDistanceFunction(const std::string &objPath, const Transformation &model, float bias);
    ObjDistanceFunction(const ObjMesh &mesh, const Transformation &model, float bias);
    ~ObjDistanceFunction() override = default;

    float distance(const glm::vec3 &p) const override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;

    bool empty() const { return triangles_.empty(); }

    // Parses an OBJ file. Returns false and fills `error` (when non-null) if
    // the file cannot be opened or contains no triangles.
    static bool loadObj(const std::string &path, ObjMesh &mesh, std::string *error = nullptr);

private:
    using Triangle = std::array<uint32_t, 3>;

    // Bounding volume hierarchy over `triangles_`; leaves index a range of
    // `order_`. Built once in the constructor and only read from distance(),
    // so concurrent octree sampling is safe.
    struct BvhNode {
        glm::vec3 min{};
        glm::vec3 max{};
        uint32_t first = 0;   // first index in order_ (leaves)
        uint32_t count = 0;   // triangle count, 0 for interior nodes
        uint32_t left = 0;
        uint32_t right = 0;
        bool isLeaf() const { return count > 0; }
    };

    enum class TriangleRegion : uint8_t {
        Face,
        EdgeAB, EdgeBC, EdgeCA,
        VertexA, VertexB, VertexC
    };

    void build(const ObjMesh &mesh, const Transformation &model, float bias);
    uint32_t buildNode(uint32_t first, uint32_t count);
    float aabbDistanceSq(const BvhNode &node, const glm::vec3 &p) const;
    const glm::vec3 &pseudoNormal(const Triangle &triangle, uint32_t triangleIndex,
                                  TriangleRegion region) const;
    static void closestPointOnTriangle(const glm::vec3 &p, const glm::vec3 &a,
                                       const glm::vec3 &b, const glm::vec3 &c,
                                       glm::vec3 &closest, TriangleRegion &region);
    static uint64_t edgeKey(uint32_t a, uint32_t b);

    std::vector<glm::vec3> vertices_;              // world-space vertices
    std::vector<Triangle> triangles_;              // indices into vertices_
    std::vector<glm::vec3> faceNormals_;           // one per triangle
    std::vector<glm::vec3> vertexNormals_;         // angle-weighted pseudo-normals
    std::vector<glm::vec3> edgeNormals_;           // indexed by edgeNormalIndex_
    std::unordered_map<uint64_t, uint32_t> edgeNormalIndex_;
    std::vector<BvhNode> nodes_;
    std::vector<uint32_t> order_;                  // triangle indices, BVH leaf order
    std::vector<glm::vec3> centroids_;
    // Largest world-space vertex distance from the model origin.
    float worldRadius_ = 0.0f;
};

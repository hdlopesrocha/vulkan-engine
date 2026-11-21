#pragma once
#include "../core/SignedDistanceFunction.hpp"
#include <array>
#include <cstdint>
#include <vector>
#include <glm/glm.hpp>
#include "../../math/geometry/Transformation.hpp"
#include "../../math/geometry/BoundingSphere.hpp"

// Local-space description of a convex polyhedron, centred on the origin.
// `vertices` should lie on (or inside) the unit sphere so the conservative
// bounding sphere below stays valid; `faces` are convex polygons listed in
// cyclic order (winding does not matter — the constructor orients them).
struct ConvexPolyhedronMesh {
    std::vector<glm::vec3> vertices;
    std::vector<std::vector<uint32_t>> faces;
};

// Exact Euclidean SDF of a convex polyhedron with cached world-space face
// triangles, mirroring TetrahedronDistanceFunction so every supported affine
// model transform (including non-uniform scale and rotation) is evaluated in
// world space.
//
// With bias = 0 the field is the exact sharp polyhedron. With a sampling bias
// (the octree leaf size) every face plane stays put and the edges/vertices are
// beveled inward by one cell, giving the trilinear lattice enough curvature to
// reproduce the silhouette instead of faceting it.
class ConvexPolyhedronDistanceFunction : public SignedDistanceFunction {
public:
    ConvexPolyhedronDistanceFunction(const Transformation &model, float bias,
                                     const ConvexPolyhedronMesh &mesh);
    virtual ~ConvexPolyhedronDistanceFunction() = default;
    float distance(const glm::vec3 &p) const override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;

    // Convex-hull face extraction for a polyhedron whose vertices surround the
    // local origin. Each face is returned as a cyclically ordered list of
    // vertex indices (one entry per face plane). Used by the regular-polyhedron
    // primitives so their meshes are derived instead of hand-tabulated.
    static std::vector<std::vector<uint32_t>> hullFaces(const std::vector<glm::vec3> &vertices);

private:
    using Triangle = std::array<glm::vec3, 3>;
    // Faces of the inset (beveled) polyhedron; their planes are the original
    // face planes.
    std::vector<Triangle> triangles_;
    // Outward normals and one point on each inset face plane.
    std::vector<glm::vec3> faceNormals_;
    std::vector<glm::vec3> facePoints_;
    float bevel_ = 0.0f;
    // Largest local-space vertex radius, for the conservative bounding sphere.
    float maxLocalRadius_ = 0.0f;
};

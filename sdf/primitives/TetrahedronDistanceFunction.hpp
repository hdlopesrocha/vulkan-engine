#pragma once
#include "../core/SignedDistanceFunction.hpp"
#include <array>
#include <glm/glm.hpp>
#include "../../math/geometry/Transformation.hpp"
#include "../../math/geometry/BoundingSphere.hpp"

// SDF of a regular tetrahedron in the conventional base-anchored orientation:
// an equilateral base on y=0 and the apex at y=1. The world-space faces and
// outward normals are cached at construction time, so the four face planes are
// exact for any affine model transform (including non-uniform scale and
// rotation).
//
// With bias = 0 the field is the exact Euclidean SDF of the sharp tetrahedron.
// When the caller passes a sampling bias (the octree leaf size), the six edges
// and four vertices are additionally beveled by min(bias/2, 6% of the smallest
// scale): the field stores enough curvature that trilinear cell reconstruction
// can resolve the silhouette instead of scalloping it.
class TetrahedronDistanceFunction : public SignedDistanceFunction {
public:
    TetrahedronDistanceFunction(const Transformation &model, float bias);
    virtual ~TetrahedronDistanceFunction() = default;
    float distance(const glm::vec3 &p) const override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;

private:
    using Triangle = std::array<glm::vec3, 3>;
    // Faces of the inset (beveled) tetrahedron.
    std::array<Triangle, 4> triangles_{};
    // Outward normals of the original face planes; the inset faces are
    // parallel to them.
    std::array<glm::vec3, 4> faceNormals_{};
    glm::vec3 centroid_{0.0f};
    float bevel_ = 0.0f;
};

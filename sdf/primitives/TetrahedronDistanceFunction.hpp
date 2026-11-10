#pragma once
#include "../core/SignedDistanceFunction.hpp"
#include <array>
#include <glm/glm.hpp>
#include "../../math/geometry/Transformation.hpp"
#include "../../math/geometry/BoundingSphere.hpp"

// Exact Euclidean SDF of a regular tetrahedron in the conventional
// base-anchored orientation: an equilateral base on y=0 and the apex at y=1.
// The world-space faces and outward normals are cached at construction time,
// so the distance is exact for any affine model transform (including
// non-uniform scale and rotation).
class TetrahedronDistanceFunction : public SignedDistanceFunction {
public:
    TetrahedronDistanceFunction(const Transformation &model, float bias);
    virtual ~TetrahedronDistanceFunction() = default;
    float distance(const glm::vec3 &p) const override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;

private:
    using Triangle = std::array<glm::vec3, 3>;
    std::array<Triangle, 4> triangles_{};
    std::array<glm::vec3, 4> faceNormals_{};
    glm::vec3 centroid_{0.0f};
};

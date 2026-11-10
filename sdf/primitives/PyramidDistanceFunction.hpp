#pragma once
#include "../SignedDistanceFunction.hpp"
#include <array>
#include <glm/glm.hpp>
#include "../../math/geometry/Transformation.hpp"
#include "../../math/geometry/BoundingSphere.hpp"

// Exact Euclidean SDF of a unit pyramid with a square base, evaluated in world
// space. The six face triangles and their outward normals are cached at
// construction time, so the distance is exact for any affine model transform
// (including non-uniform scale and rotation) instead of the old
// local-distance * min-scale lower bound.
class PyramidDistanceFunction : public SignedDistanceFunction {
public:
    PyramidDistanceFunction(const Transformation &model, float bias);
    virtual ~PyramidDistanceFunction() = default;
    float distance(const glm::vec3 &p) const override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;

private:
    using Triangle = std::array<glm::vec3, 3>;
    std::array<Triangle, 6> triangles_{};
    std::array<glm::vec3, 6> faceNormals_{};
    glm::vec3 centroid_{0.0f};
};

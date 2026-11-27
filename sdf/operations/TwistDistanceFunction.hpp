#pragma once

#include "../core/SignedDistanceFunction.hpp"
#include "../../math/geometry/Transformation.hpp"

// Twist deformation around the world Y axis through the wrapped primitive's
// center. The query point is rotated by `twistRate * localY` before the base
// field is evaluated; the returned distance is divided by a conservative
// Lipschitz factor so the geometry remains safe for sphere tracing / octree
// refinement.
class TwistDistanceFunction : public SignedDistanceFunction {
public:
    TwistDistanceFunction(const SignedDistanceFunction &function, float twistRate,
                          const Transformation &model, float bias);
    ~TwistDistanceFunction() override = default;

    float distance(const glm::vec3 &p) const override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    glm::vec3 getCenter() const override { return m_center; }

private:
    const SignedDistanceFunction &m_function;
    float m_twistRate = 0.0f;
    BoundingSphere m_sphere;
};


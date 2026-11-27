#pragma once

#include "../core/SignedDistanceFunction.hpp"
#include "../../math/geometry/Transformation.hpp"

// Bend deformation along local X: every YZ slice is rotated around the X axis
// by `bendRate * localX`. Like TwistDistanceFunction, the distance is scaled
// by a conservative Lipschitz bound.
class BendDistanceFunction : public SignedDistanceFunction {
public:
    BendDistanceFunction(const SignedDistanceFunction &function, float bendRate,
                         const Transformation &model, float bias);
    ~BendDistanceFunction() override = default;

    float distance(const glm::vec3 &p) const override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    glm::vec3 getCenter() const override { return m_center; }

private:
    const SignedDistanceFunction &m_function;
    float m_bendRate = 0.0f;
    BoundingSphere m_sphere;
};


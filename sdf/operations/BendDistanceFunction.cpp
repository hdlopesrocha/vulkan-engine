#include "BendDistanceFunction.hpp"

#include <algorithm>
#include <cmath>
#include <glm/gtc/quaternion.hpp>

BendDistanceFunction::BendDistanceFunction(const SignedDistanceFunction &function,
                                           float bendRate,
                                           const Transformation &model, float bias)
    : SignedDistanceFunction(function.getCenter(), model),
      m_function(function),
      m_bendRate(bendRate) {
    const BoundingSphere base = function.getBoundingSphere();
    m_sphere = BoundingSphere(base.center, base.radius + std::max(bias, 0.0f));
}

float BendDistanceFunction::distance(const glm::vec3 &p) const {
    const glm::vec3 q = p - m_center;
    const float angle = m_bendRate * q.x;
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    const glm::vec3 bent(q.x, c * q.y - s * q.z, s * q.y + c * q.z);

    // Rotation angle depends on x, so the Jacobian gains a rate * |yz|
    // term; dividing by it keeps the field conservative.
    const float radial = std::sqrt(q.y * q.y + q.z * q.z);
    const float lipschitz = std::max(1.0f + std::abs(m_bendRate) * radial, 1.0e-4f);
    return m_function.distance(m_center + bent) / lipschitz;
}

BoundingSphere BendDistanceFunction::getSphere(const Transformation &, float) const {
    return m_sphere;
}

ContainmentType BendDistanceFunction::check(const BoundingCube &cube) const {
    return m_sphere.test(cube);
}

bool BendDistanceFunction::isContained(const BoundingCube &cube) const {
    return cube.contains(m_sphere);
}


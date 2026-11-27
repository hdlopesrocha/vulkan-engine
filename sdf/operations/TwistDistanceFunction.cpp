#include "TwistDistanceFunction.hpp"

#include <algorithm>
#include <cmath>
#include <glm/gtc/quaternion.hpp>

TwistDistanceFunction::TwistDistanceFunction(const SignedDistanceFunction &function,
                                             float twistRate,
                                             const Transformation &model, float bias)
    : SignedDistanceFunction(function.getCenter(), model),
      m_function(function),
      m_twistRate(twistRate) {
    const BoundingSphere base = function.getBoundingSphere();
    m_sphere = BoundingSphere(base.center, base.radius + std::max(bias, 0.0f));
}

float TwistDistanceFunction::distance(const glm::vec3 &p) const {
    const glm::vec3 q = p - m_center;
    const float angle = m_twistRate * q.y;
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    const glm::vec3 twisted(c * q.x - s * q.z, q.y, s * q.x + c * q.z);

    // The twist map has |dT/dp| <= 1 + |rate| * radialDistance. Dividing by
    // this bound keeps the field conservative for sphere tracing.
    const float radial = std::sqrt(q.x * q.x + q.z * q.z);
    const float lipschitz = std::max(1.0f + std::abs(m_twistRate) * radial, 1.0e-4f);
    return m_function.distance(m_center + twisted) / lipschitz;
}

BoundingSphere TwistDistanceFunction::getSphere(const Transformation &, float) const {
    return m_sphere;
}

ContainmentType TwistDistanceFunction::check(const BoundingCube &cube) const {
    return m_sphere.test(cube);
}

bool TwistDistanceFunction::isContained(const BoundingCube &cube) const {
    return cube.contains(m_sphere);
}


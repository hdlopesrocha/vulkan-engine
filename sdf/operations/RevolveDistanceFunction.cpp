#include "RevolveDistanceFunction.hpp"

#include <algorithm>
#include <cmath>
#include <glm/gtc/quaternion.hpp>

RevolveDistanceFunction::RevolveDistanceFunction(
    std::shared_ptr<const SignedDistance2D> profile,
    const Transformation &model, float bias)
    : SignedDistanceFunction(model.translate, model), m_profile(std::move(profile)) {
    const float sx = std::max(std::abs(model.scale.x), 1.0e-4f);
    const float sy = std::max(std::abs(model.scale.y), 1.0e-4f);
    const float pr = m_profile ? std::max(m_profile->maxRadius(), 1.0e-4f) : 1.0f;
    const float radius = std::sqrt((sx * pr) * (sx * pr) + (sy * pr) * (sy * pr));
    m_sphere = BoundingSphere(model.translate, radius + std::max(bias, 0.0f));
}

float RevolveDistanceFunction::distance(const glm::vec3 &p) const {
    const glm::vec3 local = glm::inverse(m_model.quaternion) * (p - m_center);
    const float sx = std::max(std::abs(m_model.scale.x), 1.0e-4f);
    const float sy = std::max(std::abs(m_model.scale.y), 1.0e-4f);

    const float radial = std::sqrt(local.x * local.x + local.z * local.z);
    const glm::vec2 profileP(radial / sx, local.y / sy);
    const float d2 = m_profile
        ? m_profile->distance(profileP) * std::min(sx, sy)
        : 1.0e5f;
    return d2;
}

BoundingSphere RevolveDistanceFunction::getSphere(const Transformation &, float) const {
    return m_sphere;
}

ContainmentType RevolveDistanceFunction::check(const BoundingCube &cube) const {
    return m_sphere.test(cube);
}

bool RevolveDistanceFunction::isContained(const BoundingCube &cube) const {
    return cube.contains(m_sphere);
}


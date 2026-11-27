#include "ExtrudeDistanceFunction.hpp"

#include <algorithm>
#include <cmath>
#include <glm/gtc/quaternion.hpp>

ExtrudeDistanceFunction::ExtrudeDistanceFunction(
    std::shared_ptr<const SignedDistance2D> profile,
    const Transformation &model, float bias)
    : SignedDistanceFunction(model.translate, model), m_profile(std::move(profile)) {
    const float sx = std::max(std::abs(model.scale.x), 1.0e-4f);
    const float sy = std::max(std::abs(model.scale.y), 1.0e-4f);
    const float sz = std::max(std::abs(model.scale.z), 1.0e-4f);
    const float pr = m_profile ? std::max(m_profile->maxRadius(), 1.0e-4f) : 1.0f;
    const float radius = std::sqrt((sx * pr) * (sx * pr) +
                                   (sy * pr) * (sy * pr) +
                                   sz * sz);
    m_sphere = BoundingSphere(model.translate, radius + std::max(bias, 0.0f));
}

float ExtrudeDistanceFunction::distance(const glm::vec3 &p) const {
    const glm::vec3 local = glm::inverse(m_model.quaternion) * (p - m_center);
    const float sx = std::max(std::abs(m_model.scale.x), 1.0e-4f);
    const float sy = std::max(std::abs(m_model.scale.y), 1.0e-4f);
    const float sz = std::max(std::abs(m_model.scale.z), 1.0e-4f);

    // 2D profile distance (normalized), converted back to world units.
    const float d2 = m_profile
        ? m_profile->distance(glm::vec2(local.x / sx, local.y / sy)) * std::min(sx, sy)
        : 1.0e5f;
    const float dz = std::abs(local.z) - sz;

    // Box-intersection combine: exact for (extruded profile) ∩ slab.
    const float outside = std::max(std::max(d2, dz), 0.0f);
    const float inside = std::min(std::max(d2, dz), 0.0f);
    return outside + inside;
}

BoundingSphere ExtrudeDistanceFunction::getSphere(const Transformation &, float) const {
    return m_sphere;
}

ContainmentType ExtrudeDistanceFunction::check(const BoundingCube &cube) const {
    return m_sphere.test(cube);
}

bool ExtrudeDistanceFunction::isContained(const BoundingCube &cube) const {
    return cube.contains(m_sphere);
}


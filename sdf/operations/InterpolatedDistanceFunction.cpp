#include "InterpolatedDistanceFunction.hpp"

#include <algorithm>
#include <cmath>

namespace {
// Smallest sphere containing both input spheres.
BoundingSphere unionSphere(const BoundingSphere &a, const BoundingSphere &b, float extraBias) {
    const glm::vec3 delta = b.center - a.center;
    const float dist = glm::length(delta);
    if (dist + b.radius <= a.radius) {
        return BoundingSphere(a.center, a.radius + extraBias);
    }
    if (dist + a.radius <= b.radius) {
        return BoundingSphere(b.center, b.radius + extraBias);
    }
    const float radius = 0.5f * (dist + a.radius + b.radius);
    const glm::vec3 center = (dist > 1e-6f)
        ? a.center + delta * ((radius - a.radius) / dist)
        : a.center;
    return BoundingSphere(center, radius + extraBias);
}
} // namespace

InterpolatedDistanceFunction::InterpolatedDistanceFunction(
        const SignedDistanceFunction &source, const SignedDistanceFunction &target,
        float factor, const Transformation &model, float bias)
    : SignedDistanceFunction(model),
      source_(source),
      target_(target),
      factor_(std::clamp(factor, 0.0f, 1.0f)) {
    m_sphere = unionSphere(source_.getBoundingSphere(), target_.getBoundingSphere(), bias);
    m_center = m_sphere.center;
}

float InterpolatedDistanceFunction::distance(const glm::vec3 &p) const {
    return glm::mix(source_.distance(p), target_.distance(p), factor_);
}

ContainmentType InterpolatedDistanceFunction::check(const BoundingCube &cube) const {
    const ContainmentType a = source_.check(cube);
    const ContainmentType b = target_.check(cube);
    if (a == ContainmentType::Disjoint && b == ContainmentType::Disjoint) {
        return ContainmentType::Disjoint;
    }
    // Only claim containment when both operands contain the cube; a blend of
    // two fields can otherwise still cross inside it.
    if (a == ContainmentType::Contains && b == ContainmentType::Contains) {
        return ContainmentType::Contains;
    }
    return ContainmentType::Intersects;
}

bool InterpolatedDistanceFunction::isContained(const BoundingCube &cube) const {
    return source_.isContained(cube) && target_.isContained(cube);
}

BoundingSphere InterpolatedDistanceFunction::getSphere(const Transformation &model, float bias) const {
    (void)model;
    return BoundingSphere(m_sphere.center, m_sphere.radius + bias);
}

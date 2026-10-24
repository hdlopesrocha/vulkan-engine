#include "TaperedCylinderDistanceFunction.hpp"

TaperedCylinderDistanceFunction::TaperedCylinderDistanceFunction(float r1_, float r2_,
                                                                const Transformation &model, float bias)
    : SignedDistanceFunction(model.translate, model), r1(r1_), r2(r2_) {
    m_sphere = getSphere(model, bias);
}

float TaperedCylinderDistanceFunction::distance(const glm::vec3 &p) const {
    glm::vec3 pos = p - getCenter();
    pos = glm::inverse(m_model.quaternion) * pos;
    glm::vec3 q = pos / m_model.scale;

    // Capped cone: bottom radius r1 at y=-0.5, top radius r2 at y=+0.5.
    const float h = 0.5f;
    glm::vec2 w = glm::vec2(glm::length(glm::vec2(q.x, q.z)), q.y);
    glm::vec2 k1 = glm::vec2(r2, h);
    glm::vec2 k2 = glm::vec2(r2 - r1, 2.0f * h);
    glm::vec2 ca = glm::vec2(w.x - glm::min(w.x, (w.y < 0.0f) ? r1 : r2), glm::abs(w.y) - h);
    glm::vec2 cb = w - k1 + k2 * glm::clamp(glm::dot(k1 - w, k2) / glm::dot(k2, k2), 0.0f, 1.0f);
    float s = (cb.x < 0.0f && ca.y < 0.0f) ? -1.0f : 1.0f;
    float d = s * glm::sqrt(glm::min(glm::dot(ca, ca), glm::dot(cb, cb)));

    float minScale = glm::min(glm::min(m_model.scale.x, m_model.scale.y), m_model.scale.z);
    return d * minScale;
}

BoundingSphere TaperedCylinderDistanceFunction::getSphere(const Transformation &model, float bias) const {
    float maxRadius = glm::max(r1, r2);
    return BoundingSphere(getCenter(), glm::length(model.scale) * glm::sqrt(maxRadius * maxRadius + 0.25f) + bias);
}


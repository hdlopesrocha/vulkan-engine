#include "TaperedCapsuleDistanceFunction.hpp"

TaperedCapsuleDistanceFunction::TaperedCapsuleDistanceFunction(glm::vec3 a_, glm::vec3 b_, float r1_, float r2_,
                                                               const Transformation &model, float bias)
    : SignedDistanceFunction(0.5f*(a_+b_)+model.translate, model), a(a_), b(b_), r1(r1_), r2(r2_) {
    m_sphere = getSphere(model, bias);
}

float TaperedCapsuleDistanceFunction::distance(const glm::vec3 &p) const {
    glm::vec3 pos = p - m_model.translate;
    pos = glm::inverse(m_model.quaternion) * pos;
    glm::vec3 q = pos / m_model.scale;

    glm::vec3 pa = q - a, ba = b - a;
    float h = glm::clamp(glm::dot(pa, ba) / glm::dot(ba, ba), 0.0f, 1.0f);
    float r = glm::mix(r1, r2, h);
    float d = glm::length(pa - ba * h) - r;

    float minScale = glm::min(glm::min(m_model.scale.x, m_model.scale.y), m_model.scale.z);
    return d * minScale;
}

BoundingSphere TaperedCapsuleDistanceFunction::getSphere(const Transformation &model, float bias) const {
    float halfLen = glm::length(b - a) * 0.5f;
    float maxR = glm::max(r1, r2);
    glm::vec3 center = model.translate + model.quaternion * (model.scale * 0.5f * (a + b));
    return BoundingSphere(center, glm::length(model.scale) * (halfLen + maxR) + bias);
}


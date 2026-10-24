#include "CylinderDistanceFunction.hpp"

CylinderDistanceFunction::CylinderDistanceFunction(const Transformation &model, float bias)
    : SignedDistanceFunction(model.translate, model) {
    m_sphere = getSphere(model, bias);
}

float CylinderDistanceFunction::distance(const glm::vec3 &p) const {
    glm::vec3 pos = p - getCenter();
    pos = glm::inverse(m_model.quaternion) * pos;
    glm::vec3 q = pos / m_model.scale;

    // Unit cylinder: radius 0.5, half-height 1.
    glm::vec2 d = glm::vec2(glm::length(glm::vec2(q.x, q.z)) - 0.5f, glm::abs(q.y) - 1.0f);
    float dist = glm::min(glm::max(d.x, d.y), 0.0f) + glm::length(glm::max(d, 0.0f));

    float minScale = glm::min(glm::min(m_model.scale.x, m_model.scale.y), m_model.scale.z);
    return dist * minScale;
}

BoundingSphere CylinderDistanceFunction::getSphere(const Transformation &model, float bias) const {
    return isotropicSphere(getCenter(), model, bias);
}


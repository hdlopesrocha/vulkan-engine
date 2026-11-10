#include "BoxDistanceFunction.hpp"

BoxDistanceFunction::BoxDistanceFunction(const Transformation &model, float bias)
    : SignedDistanceFunction(model.translate, model) {
    m_sphere = getSphere(model, bias);
}

float BoxDistanceFunction::boxDistance(const glm::vec3 &p, const glm::vec3 &len) {
    glm::vec3 q = glm::abs(p) - len;
    return glm::length(glm::max(q, glm::vec3(0.0f)))
         + glm::min(glm::max(q.x, glm::max(q.y, q.z)), 0.0f);
}

float BoxDistanceFunction::distance(const glm::vec3 &p) const {
    glm::vec3 pos = p - getCenter();
    pos = glm::inverse(m_model.quaternion) * pos;
    return boxDistance(pos, m_model.scale);
}

BoundingSphere BoxDistanceFunction::getSphere(const Transformation &model, float bias) const {
    return BoundingSphere(getCenter(), glm::length(model.scale) + bias);
}


#include "ConeDistanceFunction.hpp"

ConeDistanceFunction::ConeDistanceFunction(const Transformation &model, float bias)
    : SignedDistanceFunction(model.translate, model) {
    m_sphere = getSphere(model, bias);
}

float ConeDistanceFunction::distance(const glm::vec3 &p) const {
    glm::vec3 pos = p - getCenter();
    pos = glm::inverse(m_model.quaternion) * pos;
    glm::vec3 q = pos / m_model.scale - glm::vec3(0,1,0);

    // Unit cone: apex at origin, base radius = height = 1.
    const glm::vec2 k(1.0f, -1.0f);
    glm::vec2 w(glm::length(glm::vec2(q.x, q.z)), q.y);
    glm::vec2 a = w - k * glm::clamp(glm::dot(w, k) / glm::dot(k, k), 0.0f, 1.0f);
    glm::vec2 b = w - k * glm::vec2(glm::clamp(w.x / k.x, 0.0f, 1.0f), 1.0f);

    float s = glm::sign(k.y);
    float dd = glm::min(glm::dot(a, a), glm::dot(b, b));
    float plane = glm::max(s * (w.x * k.y - w.y * k.x), s * (w.y - k.y));
    float d = glm::sqrt(dd) * glm::sign(plane);

    float minScale = glm::min(glm::min(m_model.scale.x, m_model.scale.y), m_model.scale.z);
    return d * minScale;
}

BoundingSphere ConeDistanceFunction::getSphere(const Transformation &model, float bias) const {
    return isotropicSphere(getCenter(), model, bias);
}


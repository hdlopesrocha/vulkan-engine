#include "OctahedronDistanceFunction.hpp"

OctahedronDistanceFunction::OctahedronDistanceFunction(const Transformation &model, float bias)
    : SignedDistanceFunction(model.translate, model) {
    m_sphere = getSphere(model, bias);
}

float OctahedronDistanceFunction::distance(const glm::vec3 &p) const {
    glm::vec3 pos = p - getCenter();
    pos = glm::inverse(m_model.quaternion) * pos;

    glm::vec3 q = pos / m_model.scale;

    // Unit octahedron (s = 1).
    glm::vec3 p2 = glm::abs(q);
    float m = p2.x + p2.y + p2.z - 1.0f;
    glm::vec3 r;
    if (3.0 * p2.x < m) r = glm::vec3(p2.x, p2.y, p2.z);
    else if (3.0 * p2.y < m) r = glm::vec3(p2.y, p2.z, p2.x);
    else if (3.0 * p2.z < m) r = glm::vec3(p2.z, p2.x, p2.y);
    else {
        float d = m * 0.57735027;
        float minScale = glm::min(glm::min(m_model.scale.x, m_model.scale.y), m_model.scale.z);
        return d * minScale;
    }

    float k = glm::clamp(0.5f * (r.z - r.y + 1.0f), 0.0f, 1.0f);
    float d = glm::length(glm::vec3(r.x, r.y - 1.0f + k, r.z - k));

    float minScale = glm::min(glm::min(m_model.scale.x, m_model.scale.y), m_model.scale.z);
    return d * minScale;
}

BoundingSphere OctahedronDistanceFunction::getSphere(const Transformation &model, float bias) const {
    return isotropicSphere(getCenter(), model, bias);
}


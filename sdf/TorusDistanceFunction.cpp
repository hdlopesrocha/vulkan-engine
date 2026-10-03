#include "TorusDistanceFunction.hpp"

TorusDistanceFunction::TorusDistanceFunction(glm::vec2 radius_, const Transformation &model, float bias)
    : SignedDistanceFunction(model.translate, model), radius(radius_) {
    m_sphere = getSphere(model, bias);
}

float TorusDistanceFunction::distance(const glm::vec3 &p) const {
     glm::vec3 pos = p - getCenter();
    pos = glm::inverse(m_model.quaternion) * pos;

    glm::vec3 q = pos / m_model.scale;
    float d = SDF::torus(q, radius);

    float minScale = glm::min(glm::min(m_model.scale.x, m_model.scale.y), m_model.scale.z);
    return d * minScale;
}

BoundingSphere TorusDistanceFunction::getSphere(const Transformation &model, float bias) const {
    return isotropicSphere(getCenter(), model, bias);
}


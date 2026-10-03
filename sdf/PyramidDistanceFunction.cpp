#include "PyramidDistanceFunction.hpp"

PyramidDistanceFunction::PyramidDistanceFunction(const Transformation &model, float bias)
    : SignedDistanceFunction(model.translate, model) {
    m_sphere = getSphere(model, bias);
}

float PyramidDistanceFunction::distance(const glm::vec3 &p) const {
   glm::vec3 pos = p - getCenter();
    pos = glm::inverse(m_model.quaternion) * pos;

    pos /= m_model.scale;

    float d = SDF::pyramid(pos, 1.0f, sqrt(0.5f));

    float minScale = glm::min(glm::min(m_model.scale.x, m_model.scale.y), m_model.scale.z);
    return d * minScale;
}

BoundingSphere PyramidDistanceFunction::getSphere(const Transformation &model, float bias) const {
    return isotropicSphere(getCenter(), model, bias);
}


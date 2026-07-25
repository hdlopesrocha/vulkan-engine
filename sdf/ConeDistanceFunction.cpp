#include "ConeDistanceFunction.hpp"
#include "SDF.hpp"

ConeDistanceFunction::ConeDistanceFunction(const Transformation &model, float bias)
    : SignedDistanceFunction(SdfType::CONE, model.translate, model)
    , sphere(getSphere(model, bias)) {}

float ConeDistanceFunction::distance(const glm::vec3 &p) const {
    glm::vec3 pos = p - getCenter();
    pos = glm::inverse(m_model.quaternion) * pos;
    glm::vec3 q = pos / m_model.scale - glm::vec3(0,1,0);

    float d = SDF::cone(q);

    float minScale = glm::min(glm::min(m_model.scale.x, m_model.scale.y), m_model.scale.z);
    return d * minScale;
}

BoundingSphere ConeDistanceFunction::getSphere(const Transformation &model, float bias) const {
    return BoundingSphere(getCenter(), sqrt(0.5f) * glm::length(model.scale) + bias);
}

ContainmentType ConeDistanceFunction::check(const BoundingCube &cube) const {
    return sphere.test(cube);
}

bool ConeDistanceFunction::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
}

const char* ConeDistanceFunction::getLabel() const {
    return "Cone";
}

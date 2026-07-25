#include "ConeDistanceFunction.hpp"
#include "SDF.hpp"

ConeDistanceFunction::ConeDistanceFunction(const Transformation &model, float bias)
    : SignedDistanceFunction(SdfType::CONE)
    , sphere(getSphere(model, bias)) {}

float ConeDistanceFunction::distance(const glm::vec3 &p, const Transformation &model) const {
    glm::vec3 pos = p - getCenter(model);
    pos = glm::inverse(model.quaternion) * pos;
    glm::vec3 q = pos / model.scale - glm::vec3(0,1,0);

    float d = SDF::cone(q);

    float minScale = glm::min(glm::min(model.scale.x, model.scale.y), model.scale.z);
    return d * minScale;
}

BoundingSphere ConeDistanceFunction::getSphere(const Transformation &model, float bias) const {
    return BoundingSphere(getCenter(model), sqrt(0.5f) * glm::length(model.scale) + bias);
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

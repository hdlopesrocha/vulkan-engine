#include "SphereDistanceFunction.hpp"

SphereDistanceFunction::SphereDistanceFunction(const Transformation &model, float bias)
    : SignedDistanceFunction(SdfType::SPHERE, model.translate, model)
    , sphere(getSphere(model, bias)) {}

float SphereDistanceFunction::distance(const glm::vec3 &p) const {
    glm::vec3 pos = p - m_model.translate;
    pos = glm::inverse(m_model.quaternion) * pos;

    glm::vec3 radii = m_model.scale;
    glm::vec3 q = glm::abs(pos) / radii;
    return (glm::length(q) - 1.0f) * glm::min(glm::min(radii.x, radii.y), radii.z);
}

BoundingSphere SphereDistanceFunction::getSphere(const Transformation &model, float bias) const {
    return BoundingSphere(getCenter(), glm::length(model.scale)*sqrt(0.5f) + bias);
}

ContainmentType SphereDistanceFunction::check(const BoundingCube &cube) const {
    return sphere.test(cube);
}

bool SphereDistanceFunction::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
}

const char* SphereDistanceFunction::getLabel() const {
    return "Sphere";
}

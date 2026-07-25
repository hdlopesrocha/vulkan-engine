#include "BoxDistanceFunction.hpp"

BoxDistanceFunction::BoxDistanceFunction(const Transformation &model, float bias)
    : SignedDistanceFunction(SdfType::BOX, model.translate, model)
    , sphere(getSphere(model, bias)) {}

float BoxDistanceFunction::distance(const glm::vec3 &p) const {
    glm::vec3 pos = p - getCenter();
    pos = glm::inverse(m_model.quaternion) * pos;
    return SDF::box(pos, m_model.scale);
}

BoundingSphere BoxDistanceFunction::getSphere(const Transformation &model, float bias) const {
    return BoundingSphere(getCenter(), glm::length(model.scale) + bias);
}

ContainmentType BoxDistanceFunction::check(const BoundingCube &cube) const {
    return sphere.test(cube);
}

bool BoxDistanceFunction::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
}

const char* BoxDistanceFunction::getLabel() const {
    return "Box";
}

#include "BoxDistanceFunction.hpp"

BoxDistanceFunction::BoxDistanceFunction(const Transformation &model, float bias)
    : SignedDistanceFunction(SdfType::BOX)
    , sphere(getSphere(model, bias)) {}

float BoxDistanceFunction::distance(const glm::vec3 &p, const Transformation &model) const {
    glm::vec3 pos = p - getCenter(model);
    pos = glm::inverse(model.quaternion) * pos;
    return SDF::box(pos, model.scale);
}

BoundingSphere BoxDistanceFunction::getSphere(const Transformation &model, float bias) const {
    return BoundingSphere(getCenter(model), glm::length(model.scale) + bias);
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

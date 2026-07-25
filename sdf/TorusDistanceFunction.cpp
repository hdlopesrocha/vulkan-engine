#include "TorusDistanceFunction.hpp"

TorusDistanceFunction::TorusDistanceFunction(glm::vec2 radius_, const Transformation &model, float bias)
    : SignedDistanceFunction(SdfType::TORUS, model.translate), radius(radius_), sphere(getSphere(model, bias)) {}

float TorusDistanceFunction::distance(const glm::vec3 &p, const Transformation &model) const {
     glm::vec3 pos = p - getCenter();
    pos = glm::inverse(model.quaternion) * pos;

    glm::vec3 q = pos / model.scale;
    float d = SDF::torus(q, radius);

    float minScale = glm::min(glm::min(model.scale.x, model.scale.y), model.scale.z);
    return d * minScale;
}

BoundingSphere TorusDistanceFunction::getSphere(const Transformation &model, float bias) const {
    return BoundingSphere(getCenter(), glm::length(model.scale)*sqrt(0.5f) + bias);
}

ContainmentType TorusDistanceFunction::check(const BoundingCube &cube) const {
    return sphere.test(cube);
}

bool TorusDistanceFunction::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
}

const char* TorusDistanceFunction::getLabel() const {
    return "Torus";
}

#include "CapsuleDistanceFunction.hpp"

CapsuleDistanceFunction::CapsuleDistanceFunction(glm::vec3 a_, glm::vec3 b_, float r, const Transformation &model, float bias)
    : SignedDistanceFunction(SdfType::CAPSULE), a(a_), b(b_), radius(r), sphere(getSphere(model, bias)) {}

float CapsuleDistanceFunction::distance(const glm::vec3 &p, const Transformation &model) {
    glm::vec3 pos = p - model.translate;
    pos = glm::inverse(model.quaternion) * pos;
    float d = SDF::capsule(pos / model.scale, a, b, radius);
    float minScale = glm::min(glm::min(model.scale.x, model.scale.y), model.scale.z);
    return d * minScale;
}

glm::vec3 CapsuleDistanceFunction::getCenter(const Transformation &model) const {
    return 0.5f*(this->a+this->b)+model.translate;
}

BoundingSphere CapsuleDistanceFunction::getSphere(const Transformation &model, float bias) const {
    float halfLen = glm::length(b - a) * 0.5f;
    float maxR = radius;
    glm::vec3 center = model.translate + model.quaternion * (model.scale * 0.5f * (a + b));
    return BoundingSphere(center, glm::length(model.scale) * (halfLen + maxR) + bias);
}

ContainmentType CapsuleDistanceFunction::check(const BoundingCube &cube) const {
    return sphere.test(cube);
}

bool CapsuleDistanceFunction::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
}

const char* CapsuleDistanceFunction::getLabel() const {
    return "Capsule";
}

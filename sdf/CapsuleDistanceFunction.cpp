#include "CapsuleDistanceFunction.hpp"

CapsuleDistanceFunction::CapsuleDistanceFunction(glm::vec3 a, glm::vec3 b, float r) {
    this->a = a;
    this->b = b;
    this->radius = r;
}

float CapsuleDistanceFunction::distance(const glm::vec3 &p, const Transformation &model) {
    glm::vec3 pos = p - model.translate; // Move point into model space
    pos = glm::inverse(model.quaternion) * pos;
    return SDF::capsule(pos/model.scale, a, b, radius);
}

SdfType CapsuleDistanceFunction::getType() const {
    return SdfType::CAPSULE;
}

glm::vec3 CapsuleDistanceFunction::getCenter(const Transformation &model) const {
    return 0.5f*(this->a+this->b)+model.translate;
}


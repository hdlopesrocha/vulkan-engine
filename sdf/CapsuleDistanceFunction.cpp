#include "CapsuleDistanceFunction.hpp"

CapsuleDistanceFunction::CapsuleDistanceFunction(glm::vec3 a_, glm::vec3 b_, float r)
    : SignedDistanceFunction(SdfType::CAPSULE), a(a_), b(b_), radius(r) {}

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


#include "TaperedCapsuleDistanceFunction.hpp"
#include "SDF.hpp"

TaperedCapsuleDistanceFunction::TaperedCapsuleDistanceFunction(glm::vec3 a_, glm::vec3 b_, float r1_, float r2_,
                                                               const Transformation &model, float bias)
    : SignedDistanceFunction(SdfType::TAPERED_CAPSULE, 0.5f*(a_+b_)+model.translate), a(a_), b(b_), r1(r1_), r2(r2_)
    , sphere(getSphere(model, bias)) {}

float TaperedCapsuleDistanceFunction::distance(const glm::vec3 &p, const Transformation &model) const {
    glm::vec3 pos = p - model.translate;
    pos = glm::inverse(model.quaternion) * pos;
    float d = SDF::taperedCapsule(pos / model.scale, a, b, r1, r2);
    float minScale = glm::min(glm::min(model.scale.x, model.scale.y), model.scale.z);
    return d * minScale;
}

BoundingSphere TaperedCapsuleDistanceFunction::getSphere(const Transformation &model, float bias) const {
    float halfLen = glm::length(b - a) * 0.5f;
    float maxR = glm::max(r1, r2);
    glm::vec3 center = model.translate + model.quaternion * (model.scale * 0.5f * (a + b));
    return BoundingSphere(center, glm::length(model.scale) * (halfLen + maxR) + bias);
}

ContainmentType TaperedCapsuleDistanceFunction::check(const BoundingCube &cube) const {
    return sphere.test(cube);
}

bool TaperedCapsuleDistanceFunction::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
}

const char* TaperedCapsuleDistanceFunction::getLabel() const {
    return "Tapered Capsule";
}

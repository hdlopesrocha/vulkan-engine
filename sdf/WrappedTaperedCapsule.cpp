#include "WrappedTaperedCapsule.hpp"

WrappedTaperedCapsule::WrappedTaperedCapsule(TaperedCapsuleDistanceFunction * function_, const Transformation &model, float bias)
    : WrappedSignedDistanceFunction(function_)
    , sphere(getSphere(model, bias)) {
}

WrappedTaperedCapsule::~WrappedTaperedCapsule() {
}

BoundingSphere WrappedTaperedCapsule::getSphere(const Transformation &model, float bias) const {
    TaperedCapsuleDistanceFunction * f = (TaperedCapsuleDistanceFunction*) function;
    float halfLen = glm::length(f->b - f->a) * 0.5f;
    float maxR = glm::max(f->r1, f->r2);
    glm::vec3 center = model.translate + model.quaternion * (model.scale * 0.5f * (f->a + f->b));
    return BoundingSphere(center, glm::length(model.scale) * (halfLen + maxR) + bias);
}

ContainmentType WrappedTaperedCapsule::check(const BoundingCube &cube) const {
    return sphere.test(cube);
}

bool WrappedTaperedCapsule::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
}

const char* WrappedTaperedCapsule::getLabel() const {
    return "Tapered Capsule";
}

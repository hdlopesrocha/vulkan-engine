#include "WrappedTaperedCapsule.hpp"

WrappedTaperedCapsule::WrappedTaperedCapsule(TaperedCapsuleDistanceFunction * function_)
    : WrappedSignedDistanceFunction(function_) {
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

ContainmentType WrappedTaperedCapsule::check(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingSphere sphere = getSphere(model, bias);
    return sphere.test(cube);
}

bool WrappedTaperedCapsule::isContained(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingSphere sphere = getSphere(model, bias);
    return cube.contains(sphere);
}

void WrappedTaperedCapsule::accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const {
    getSphere(model, bias).accept(visitor);
}

const char* WrappedTaperedCapsule::getLabel() const {
    return "Tapered Capsule";
}

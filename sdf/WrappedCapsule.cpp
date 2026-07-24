#include "WrappedCapsule.hpp"

WrappedCapsule::WrappedCapsule(CapsuleDistanceFunction * function_) : WrappedSignedDistanceFunction(function_) {

}

WrappedCapsule::~WrappedCapsule() {

}

BoundingSphere WrappedCapsule::getSphere(const Transformation &model, float bias) const {
    CapsuleDistanceFunction * f = (CapsuleDistanceFunction*) function;
    float halfLen = glm::length(f->b - f->a) * 0.5f;
    float maxR = f->radius;
    glm::vec3 center = model.translate + model.quaternion * (model.scale * 0.5f * (f->a + f->b));
    return BoundingSphere(center, glm::length(model.scale) * (halfLen + maxR) + bias);
}

ContainmentType WrappedCapsule::check(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingSphere sphere = getSphere(model, bias);
    return sphere.test(cube);
}

bool WrappedCapsule::isContained(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingSphere sphere = getSphere(model, bias);
    return cube.contains(sphere);
};

void WrappedCapsule::accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const {
    getSphere(model, bias).accept(visitor);
}

const char* WrappedCapsule::getLabel() const {
    return "Capsule";
}
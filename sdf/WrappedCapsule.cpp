#include "WrappedCapsule.hpp"

WrappedCapsule::WrappedCapsule(CapsuleDistanceFunction * function_, const Transformation &model, float bias)
    : WrappedSignedDistanceFunction(function_)
    , sphere(getSphere(model, bias)) {

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

ContainmentType WrappedCapsule::check(const BoundingCube &cube) const {
    return sphere.test(cube);
}

bool WrappedCapsule::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
};

const char* WrappedCapsule::getLabel() const {
    return "Capsule";
}
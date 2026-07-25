#include "WrappedTaperedCylinder.hpp"

WrappedTaperedCylinder::WrappedTaperedCylinder(TaperedCylinderDistanceFunction * function_, const Transformation &model, float bias)
    : WrappedSignedDistanceFunction(function_)
    , sphere(getSphere(model, bias)) {
}

WrappedTaperedCylinder::~WrappedTaperedCylinder() {
}

BoundingSphere WrappedTaperedCylinder::getSphere(const Transformation &model, float bias) const {
    TaperedCylinderDistanceFunction * f = (TaperedCylinderDistanceFunction*) function;
    // Use the larger of the two radii for the bounding sphere
    float maxRadius = glm::max(f->r1, f->r2);
    return BoundingSphere(f->getCenter(model), glm::length(model.scale) * glm::sqrt(maxRadius * maxRadius + 0.25f) + bias);
}

ContainmentType WrappedTaperedCylinder::check(const BoundingCube &cube) const {
    return sphere.test(cube);
}

bool WrappedTaperedCylinder::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
}

const char* WrappedTaperedCylinder::getLabel() const {
    return "Tapered Cylinder";
}

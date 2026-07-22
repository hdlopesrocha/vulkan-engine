#include "WrappedTaperedCylinder.hpp"

WrappedTaperedCylinder::WrappedTaperedCylinder(TaperedCylinderDistanceFunction * function_)
    : WrappedSignedDistanceFunction(function_) {
}

WrappedTaperedCylinder::~WrappedTaperedCylinder() {
}

BoundingSphere WrappedTaperedCylinder::getSphere(const Transformation &model, float bias) const {
    TaperedCylinderDistanceFunction * f = (TaperedCylinderDistanceFunction*) function;
    // Use the larger of the two radii for the bounding sphere
    float maxRadius = glm::max(f->r1, f->r2);
    return BoundingSphere(f->getCenter(model), glm::length(model.scale) * glm::sqrt(maxRadius * maxRadius + 0.25f) + bias);
}

ContainmentType WrappedTaperedCylinder::check(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingSphere sphere = getSphere(model, bias);
    return sphere.test(cube);
}

bool WrappedTaperedCylinder::isContained(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingSphere sphere = getSphere(model, bias);
    return cube.contains(sphere);
}

float WrappedTaperedCylinder::getLength(const Transformation &model, float bias) const {
    return glm::length(model.scale) + bias;
}

void WrappedTaperedCylinder::accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const {
    getSphere(model, bias).accept(visitor);
}

const char* WrappedTaperedCylinder::getLabel() const {
    return "Tapered Cylinder";
}

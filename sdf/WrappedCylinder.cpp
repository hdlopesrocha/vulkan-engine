#include "WrappedCylinder.hpp"


WrappedCylinder::WrappedCylinder(CylinderDistanceFunction * function) : WrappedSignedDistanceFunction(function) {

}

WrappedCylinder::~WrappedCylinder() {

}

BoundingSphere WrappedCylinder::getSphere(const Transformation &model, float bias) const {
    CylinderDistanceFunction * f = (CylinderDistanceFunction*) function;
    return BoundingSphere(f->getCenter(model), glm::length(model.scale)*sqrt(0.5f) + bias);
};

ContainmentType WrappedCylinder::check(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingSphere sphere = getSphere(model, bias);
    return sphere.test(cube);
};

bool WrappedCylinder::isContained(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingSphere sphere = getSphere(model, bias);
    return cube.contains(sphere);
};

float WrappedCylinder::getLength(const Transformation &model, float bias) const {
    return glm::length(model.scale) + bias;
};

void WrappedCylinder::accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const {
    getSphere(model, bias).accept(visitor);
}

const char* WrappedCylinder::getLabel() const {
    return "Cylinder";
}
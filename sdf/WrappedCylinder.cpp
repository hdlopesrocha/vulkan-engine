#include "WrappedCylinder.hpp"


WrappedCylinder::WrappedCylinder(CylinderDistanceFunction * function_, const Transformation &model, float bias)
    : WrappedSignedDistanceFunction(function_)
    , sphere(getSphere(model, bias)) {

}

WrappedCylinder::~WrappedCylinder() {

}

BoundingSphere WrappedCylinder::getSphere(const Transformation &model, float bias) const {
    CylinderDistanceFunction * f = (CylinderDistanceFunction*) function;
    return BoundingSphere(f->getCenter(model), glm::length(model.scale)*sqrt(0.5f) + bias);
};

ContainmentType WrappedCylinder::check(const BoundingCube &cube) const {
    return sphere.test(cube);
};

bool WrappedCylinder::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
};


const char* WrappedCylinder::getLabel() const {
    return "Cylinder";
}
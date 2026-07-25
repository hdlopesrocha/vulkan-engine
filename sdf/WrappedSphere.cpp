#include "WrappedSphere.hpp"

WrappedSphere::WrappedSphere(SphereDistanceFunction * function_, const Transformation &model, float bias)
    : WrappedSignedDistanceFunction(function_)
    , sphere(getSphere(model, bias)) {

}

WrappedSphere::~WrappedSphere() {

}

BoundingSphere WrappedSphere::getSphere(const Transformation &model, float bias) const {
    SphereDistanceFunction * f = (SphereDistanceFunction*) function;
    return BoundingSphere(f->getCenter(model), glm::length(model.scale)*sqrt(0.5f) + bias);
};

ContainmentType WrappedSphere::check(const BoundingCube &cube) const {
    return sphere.test(cube);
};

bool WrappedSphere::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
};

const char* WrappedSphere::getLabel() const {
    return "Sphere";
}
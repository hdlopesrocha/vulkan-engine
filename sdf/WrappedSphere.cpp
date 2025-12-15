#include "WrappedSphere.hpp"

WrappedSphere::WrappedSphere(SphereDistanceFunction * function) : WrappedSignedDistanceFunction(function) {

}

WrappedSphere::~WrappedSphere() {

}

BoundingSphere WrappedSphere::getSphere(const Transformation &model, float bias) const {
    SphereDistanceFunction * f = (SphereDistanceFunction*) function;
    return BoundingSphere(f->getCenter(model), glm::length(model.scale)*sqrt(0.5f) + bias);
};

ContainmentType WrappedSphere::check(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingSphere sphere = getSphere(model, bias);
    return sphere.test(cube);
};

bool WrappedSphere::isContained(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingSphere sphere = getSphere(model, bias);
    return cube.contains(sphere);
};

float WrappedSphere::getLength(const Transformation &model, float bias) const {
    return glm::length(model.scale) + bias;
};

void WrappedSphere::accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const {
    getSphere(model, bias).accept(visitor);
}

const char* WrappedSphere::getLabel() const {
    return "Sphere";
}
#include "WrappedCone.hpp"

WrappedCone::WrappedCone(ConeDistanceFunction * function) : WrappedSignedDistanceFunction(function) {

}

WrappedCone::~WrappedCone() {

}

BoundingSphere WrappedCone::getSphere(const Transformation &model, float bias) const {
    ConeDistanceFunction * f = (ConeDistanceFunction*) function;
    return BoundingSphere(f->getCenter(model), sqrt(0.5f) * glm::length(model.scale) + bias);
};

ContainmentType WrappedCone::check(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingSphere sphere = getSphere(model, bias);
    return sphere.test(cube);
};

bool WrappedCone::isContained(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingSphere sphere = getSphere(model, bias);
    return cube.contains(sphere);
};

float WrappedCone::getLength(const Transformation &model, float bias) const {
    return glm::length(model.scale) + bias;
};

void WrappedCone::accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const {
    getSphere(model, bias).accept(visitor);
}

const char* WrappedCone::getLabel() const {
    return "Cone";
}
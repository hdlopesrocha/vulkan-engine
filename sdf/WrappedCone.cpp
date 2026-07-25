#include "WrappedCone.hpp"

WrappedCone::WrappedCone(ConeDistanceFunction * function_, const Transformation &model, float bias)
    : WrappedSignedDistanceFunction(function_)
    , sphere(getSphere(model, bias)) {

}

WrappedCone::~WrappedCone() {

}

BoundingSphere WrappedCone::getSphere(const Transformation &model, float bias) const {
    ConeDistanceFunction * f = (ConeDistanceFunction*) function;
    return BoundingSphere(f->getCenter(model), sqrt(0.5f) * glm::length(model.scale) + bias);
};

ContainmentType WrappedCone::check(const BoundingCube &cube) const {
    return sphere.test(cube);
};

bool WrappedCone::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
};

const char* WrappedCone::getLabel() const {
    return "Cone";
}
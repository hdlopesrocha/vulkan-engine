#include "WrappedTorus.hpp"

WrappedTorus::WrappedTorus(TorusDistanceFunction * function_, const Transformation &model, float bias)
    : WrappedSignedDistanceFunction(function_)
    , sphere(getSphere(model, bias)) {

}

WrappedTorus::~WrappedTorus() {

}

BoundingSphere WrappedTorus::getSphere(const Transformation &model, float bias) const {
    TorusDistanceFunction * f = (TorusDistanceFunction*) function;
    return BoundingSphere(f->getCenter(model), glm::length(model.scale)*sqrt(0.5f) + bias);
};

ContainmentType WrappedTorus::check(const BoundingCube &cube) const {
    return sphere.test(cube);
};

bool WrappedTorus::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
};

const char* WrappedTorus::getLabel() const {
    return "Torus";
}
#include "WrappedTorus.hpp"

WrappedTorus::WrappedTorus(TorusDistanceFunction * function) : WrappedSignedDistanceFunction(function) {

}

WrappedTorus::~WrappedTorus() {

}

BoundingSphere WrappedTorus::getSphere(const Transformation &model, float bias) const {
    TorusDistanceFunction * f = (TorusDistanceFunction*) function;
    return BoundingSphere(f->getCenter(model), glm::length(model.scale)*sqrt(0.5f) + bias);
};

ContainmentType WrappedTorus::check(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingSphere sphere = getSphere(model, bias);
    return sphere.test(cube);
};

bool WrappedTorus::isContained(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingSphere sphere = getSphere(model, bias);
    return cube.contains(sphere);
};

float WrappedTorus::getLength(const Transformation &model, float bias) const {
    return glm::length(model.scale) + bias;
};

void WrappedTorus::accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const {
    getSphere(model, bias).accept(visitor);
}

const char* WrappedTorus::getLabel() const {
    return "Torus";
}
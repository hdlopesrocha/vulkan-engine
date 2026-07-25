#include "WrappedOctahedron.hpp"


WrappedOctahedron::WrappedOctahedron(OctahedronDistanceFunction * function_, const Transformation &model, float bias)
    : WrappedSignedDistanceFunction(function_)
    , sphere(getSphere(model, bias)) {

}

WrappedOctahedron::~WrappedOctahedron() {

}

BoundingSphere WrappedOctahedron::getSphere(const Transformation &model, float bias) const {
    OctahedronDistanceFunction * f = (OctahedronDistanceFunction*) function;
    return BoundingSphere(f->getCenter(model), glm::length(model.scale)*sqrt(0.5f) + bias);
};

ContainmentType WrappedOctahedron::check(const BoundingCube &cube) const {
    return sphere.test(cube);
};

bool WrappedOctahedron::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
};

const char* WrappedOctahedron::getLabel() const {
    return "Octahedron";
}
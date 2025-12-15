#include "WrappedOctahedron.hpp"


WrappedOctahedron::WrappedOctahedron(OctahedronDistanceFunction * function) : WrappedSignedDistanceFunction(function) {

}

WrappedOctahedron::~WrappedOctahedron() {

}

BoundingSphere WrappedOctahedron::getSphere(const Transformation &model, float bias) const {
    OctahedronDistanceFunction * f = (OctahedronDistanceFunction*) function;
    return BoundingSphere(f->getCenter(model), glm::length(model.scale)*sqrt(0.5f) + bias);
};

ContainmentType WrappedOctahedron::check(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingSphere sphere = getSphere(model, bias);
    return sphere.test(cube);
};

bool WrappedOctahedron::isContained(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingSphere sphere = getSphere(model, bias);
    return cube.contains(sphere);
};

float WrappedOctahedron::getLength(const Transformation &model, float bias) const {
    return glm::length(model.scale) + bias;
};

void WrappedOctahedron::accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const {
    getSphere(model, bias).accept(visitor);
}    

const char* WrappedOctahedron::getLabel() const {
    return "Octahedron";
}
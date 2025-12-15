#include "WrappedPyramid.hpp"


WrappedPyramid::WrappedPyramid(PyramidDistanceFunction * function) : WrappedSignedDistanceFunction(function) {

}

WrappedPyramid::~WrappedPyramid() {

}

float WrappedPyramid::boundingSphereRadius(float width, float depth, float height) const {
    return glm::length(glm::vec3(width, height, depth));
}

BoundingSphere WrappedPyramid::getSphere(const Transformation &model, float bias) const {
    PyramidDistanceFunction * f = (PyramidDistanceFunction*) function;
    return BoundingSphere(f->getCenter(model), sqrt(0.5f) * glm::length(model.scale) + bias);
};

ContainmentType WrappedPyramid::check(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingSphere sphere = getSphere(model, bias);
    return sphere.test(cube);
};

bool WrappedPyramid::isContained(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingSphere sphere = getSphere(model, bias);
    return cube.contains(sphere);
};
float WrappedPyramid::getLength(const Transformation &model, float bias) const {
    return glm::length(model.scale) + bias;
};

void WrappedPyramid::accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const {
    getSphere(model, bias).accept(visitor);
}    

const char* WrappedPyramid::getLabel() const {
    return "Pyramid";
}
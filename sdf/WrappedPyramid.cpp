#include "WrappedPyramid.hpp"


WrappedPyramid::WrappedPyramid(PyramidDistanceFunction * function_, const Transformation &model, float bias)
    : WrappedSignedDistanceFunction(function_)
    , sphere(getSphere(model, bias)) {

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

ContainmentType WrappedPyramid::check(const BoundingCube &cube) const {
    return sphere.test(cube);
};

bool WrappedPyramid::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
};


const char* WrappedPyramid::getLabel() const {
    return "Pyramid";
}
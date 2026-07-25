#include "WrappedBox.hpp"

WrappedBox::WrappedBox(BoxDistanceFunction * function_, const Transformation &model, float bias)
    : WrappedSignedDistanceFunction(function_)
    , sphere(getSphere(model, bias)) {

}

WrappedBox::~WrappedBox() {

}

BoundingSphere WrappedBox::getSphere(const Transformation &model, float bias) const {
    BoxDistanceFunction * f = (BoxDistanceFunction*) function;
    return BoundingSphere(f->getCenter(model), glm::length(model.scale)+ bias);
};

ContainmentType WrappedBox::check(const BoundingCube &cube) const {
    return sphere.test(cube);
};

bool WrappedBox::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
};

const char* WrappedBox::getLabel() const {
    return "Box";
}
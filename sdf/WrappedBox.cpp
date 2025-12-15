#include "WrappedBox.hpp"

WrappedBox::WrappedBox(BoxDistanceFunction * function) : WrappedSignedDistanceFunction(function) {

}

WrappedBox::~WrappedBox() {

}

BoundingSphere WrappedBox::getSphere(const Transformation &model, float bias) const {
    BoxDistanceFunction * f = (BoxDistanceFunction*) function;
    return BoundingSphere(f->getCenter(model), glm::length(model.scale)+ bias);
};

ContainmentType WrappedBox::check(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingSphere sphere = getSphere(model, bias);
    return sphere.test(cube);
};

bool WrappedBox::isContained(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingSphere sphere = getSphere(model, bias);
    return cube.contains(sphere);
};

float WrappedBox::getLength(const Transformation &model, float bias) const {
    return glm::length(model.scale) + bias;
};

void WrappedBox::accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const {
    getSphere(model, bias).accept(visitor);
}

const char* WrappedBox::getLabel() const {
    return "Box";
}
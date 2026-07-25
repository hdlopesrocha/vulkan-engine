#include "WrappedSignedDistanceEffect.hpp"

WrappedSignedDistanceEffect::WrappedSignedDistanceEffect(SignedDistanceFunction * function_, const Transformation &model, float bias)
: SignedDistanceFunction(), function(function_) {
    if(function) {
        sphere = function->getSphere(model, bias);
    }
}

WrappedSignedDistanceEffect::~WrappedSignedDistanceEffect() = default;

void WrappedSignedDistanceEffect::setFunction(SignedDistanceFunction * function_) {
    this->function = function_;
}

ContainmentType WrappedSignedDistanceEffect::check(const BoundingCube &cube) const {
    return sphere.test(cube);
}

bool WrappedSignedDistanceEffect::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
}

glm::vec3 WrappedSignedDistanceEffect::getCenter(const Transformation &model) const {
    return function ? function->getCenter(model) : SignedDistanceFunction::getCenter(model);
}

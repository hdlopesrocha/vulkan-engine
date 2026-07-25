#include "SignedDistanceEffect.hpp"

SignedDistanceEffect::SignedDistanceEffect(SignedDistanceFunction * function_, const Transformation &model, float bias)
: SignedDistanceFunction(), function(function_) {
    if(function) {
        sphere = function->getSphere(model, bias);
    }
}

SignedDistanceEffect::~SignedDistanceEffect() = default;

void SignedDistanceEffect::setFunction(SignedDistanceFunction * function_) {
    this->function = function_;
}

ContainmentType SignedDistanceEffect::check(const BoundingCube &cube) const {
    return sphere.test(cube);
}

bool SignedDistanceEffect::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
}

glm::vec3 SignedDistanceEffect::getCenter(const Transformation &model) const {
    return function ? function->getCenter(model) : SignedDistanceFunction::getCenter(model);
}

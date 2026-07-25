#include "WrappedSignedDistanceEffect.hpp"

WrappedSignedDistanceEffect::WrappedSignedDistanceEffect(WrappedSignedDistanceFunction * function_, const Transformation &model, float bias)
: WrappedSignedDistanceFunction(function_) {
    auto wf = dynamic_cast<WrappedSignedDistanceFunction*>(function);
    if(wf) {
        sphere = wf->getSphere(model, bias);
    }
}

WrappedSignedDistanceEffect::~WrappedSignedDistanceEffect() = default;

void WrappedSignedDistanceEffect::setFunction(WrappedSignedDistanceFunction * function_) {
    this->function = function_;
}

ContainmentType WrappedSignedDistanceEffect::check(const BoundingCube &cube) const {
    return sphere.test(cube);
}

bool WrappedSignedDistanceEffect::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
}

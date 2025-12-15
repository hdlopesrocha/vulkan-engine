#include "WrappedSignedDistanceEffect.hpp"

WrappedSignedDistanceEffect::WrappedSignedDistanceEffect(WrappedSignedDistanceFunction * function)
: WrappedSignedDistanceFunction(function) {}

WrappedSignedDistanceEffect::~WrappedSignedDistanceEffect() = default;

void WrappedSignedDistanceEffect::setFunction(WrappedSignedDistanceFunction * function) {
    this->function = function;
}

ContainmentType WrappedSignedDistanceEffect::check(const BoundingCube &cube, const Transformation &model, float bias) const {
    auto wf = dynamic_cast<WrappedSignedDistanceFunction*>(function);
    if(wf) return wf->check(cube, model, bias);
    return Intersects;
}

bool WrappedSignedDistanceEffect::isContained(const BoundingCube &cube, const Transformation &model, float bias) const {
    auto wf = dynamic_cast<WrappedSignedDistanceFunction*>(function);
    if(wf) return wf->isContained(cube, model, bias);
    return false;
}

float WrappedSignedDistanceEffect::getLength(const Transformation &model, float bias) const {
    auto wf = dynamic_cast<WrappedSignedDistanceFunction*>(function);
    if(wf) return wf->getLength(model, bias);
    return 0.0f;
}

void WrappedSignedDistanceEffect::accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const {
    auto wf = dynamic_cast<WrappedSignedDistanceFunction*>(function);
    if(wf) wf->accept(visitor, model, bias);
}


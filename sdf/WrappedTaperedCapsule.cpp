#include "WrappedTaperedCapsule.hpp"

WrappedTaperedCapsule::WrappedTaperedCapsule(TaperedCapsuleDistanceFunction * function)
    : WrappedSignedDistanceFunction(function) {
}

WrappedTaperedCapsule::~WrappedTaperedCapsule() {
}

BoundingBox WrappedTaperedCapsule::getBox(const Transformation &model, float bias) const {
    TaperedCapsuleDistanceFunction * f = (TaperedCapsuleDistanceFunction*) function;
    float s = glm::length(model.scale);
    glm::vec3 min = glm::min(f->a, f->b) * s + model.translate;
    glm::vec3 max = glm::max(f->a, f->b) * s + model.translate;
    float maxRadius = glm::max(f->r1, f->r2) * s;
    glm::vec3 len = glm::vec3(maxRadius + bias);
    return BoundingBox(min - len, max + len);
}

ContainmentType WrappedTaperedCapsule::check(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingBox box = getBox(model, bias);
    return box.test(cube);
}

bool WrappedTaperedCapsule::isContained(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingBox box = getBox(model, bias);
    return cube.contains(box);
}

float WrappedTaperedCapsule::getLength(const Transformation &model, float bias) const {
    BoundingBox box = getBox(model, bias);
    return glm::distance(box.getMin(), box.getMax());
}

void WrappedTaperedCapsule::accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const {
    getBox(model, bias).accept(visitor);
}

const char* WrappedTaperedCapsule::getLabel() const {
    return "Tapered Capsule";
}

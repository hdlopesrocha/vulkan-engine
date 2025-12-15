#include "WrappedOctreeDifference.hpp"

WrappedOctreeDifference::WrappedOctreeDifference(OctreeDifferenceFunction * function) : WrappedSignedDistanceFunction(function) {

}

WrappedOctreeDifference::~WrappedOctreeDifference() {

}

BoundingBox WrappedOctreeDifference::getBox(float bias) const {
    OctreeDifferenceFunction * f = (OctreeDifferenceFunction*) function;
    return BoundingBox(f->box.getMin()-glm::vec3(bias), f->box.getMax()+glm::vec3(bias));
}
    
ContainmentType WrappedOctreeDifference::check(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingBox box = getBox(bias);
    return box.test(cube);
};

bool WrappedOctreeDifference::isContained(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingBox box = getBox(bias);
    return cube.contains(box);
};

glm::vec3 WrappedOctreeDifference::getCenter(const Transformation &model) const {
    OctreeDifferenceFunction * f = (OctreeDifferenceFunction*) function;
    return f->box.getCenter();
};
float WrappedOctreeDifference::getLength(const Transformation &model, float bias) const {
    BoundingBox box = getBox(bias);
    return glm::length(box.getLength()) + bias;
};

void WrappedOctreeDifference::accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const {
    getBox(bias).accept(visitor);
}

const char* WrappedOctreeDifference::getLabel() const {
    return "Octree Difference";
}


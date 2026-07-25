#include "WrappedOctreeDifference.hpp"

WrappedOctreeDifference::WrappedOctreeDifference(OctreeDifferenceFunction * function_, float bias)
    : WrappedSignedDistanceFunction(function_)
    , box(getBox(bias)) {

}

WrappedOctreeDifference::~WrappedOctreeDifference() {

}

BoundingBox WrappedOctreeDifference::getBox(float bias) const {
    OctreeDifferenceFunction * f = (OctreeDifferenceFunction*) function;
    return BoundingBox(f->box.getMin()-glm::vec3(bias), f->box.getMax()+glm::vec3(bias));
}
    
ContainmentType WrappedOctreeDifference::check(const BoundingCube &cube) const {
    return box.test(cube);
};

bool WrappedOctreeDifference::isContained(const BoundingCube &cube) const {
    return cube.contains(box);
};

glm::vec3 WrappedOctreeDifference::getCenter(const Transformation &model) const {
    OctreeDifferenceFunction * f = (OctreeDifferenceFunction*) function;
    return f->box.getCenter();
};

const char* WrappedOctreeDifference::getLabel() const {
    return "Octree Difference";
}


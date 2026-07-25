#include "OctreeDifferenceFunction.hpp"

OctreeDifferenceFunction::OctreeDifferenceFunction(Octree * tree_, BoundingBox box_, float bias_, const Transformation &model)
    : SignedDistanceFunction(SdfType::OCTREE_DIFFERENCE, box_.getCenter(), model), tree(tree_), box(box_), bias(bias_)
    , m_box(getBox(bias_)) {}

float OctreeDifferenceFunction::distance(const glm::vec3 &p) const {
    glm::vec3 len = box.getLength()*0.5f;
    glm::vec3 pos = p - box.getCenter() + m_model.translate;
    return SDF::opSubtraction(
        SDF::box(pos, len),
        tree->getSdfAt(p)+bias
    );
}

glm::vec3 OctreeDifferenceFunction::getCenter() const {
    return m_center;
}

BoundingBox OctreeDifferenceFunction::getBox(float bias_) const {
    return BoundingBox(box.getMin()-glm::vec3(bias_), box.getMax()+glm::vec3(bias_));
}

ContainmentType OctreeDifferenceFunction::check(const BoundingCube &cube) const {
    return m_box.test(cube);
}

bool OctreeDifferenceFunction::isContained(const BoundingCube &cube) const {
    return cube.contains(m_box);
}

const char* OctreeDifferenceFunction::getLabel() const {
    return "Octree Difference";
}

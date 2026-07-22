#include "OctreeDifferenceFunction.hpp"

OctreeDifferenceFunction::OctreeDifferenceFunction(Octree * tree_, BoundingBox box_, float bias_)
    : SignedDistanceFunction(SdfType::OCTREE_DIFFERENCE), tree(tree_), box(box_), bias(bias_) {}

float OctreeDifferenceFunction::distance(const glm::vec3 &p, const Transformation &model) {
    glm::vec3 len = box.getLength()*0.5f;
    glm::vec3 pos = p - box.getCenter()+model.translate;
    return SDF::opSubtraction(
        SDF::box(pos, len),
        tree->getSdfAt(p)+bias
    );
}

glm::vec3 OctreeDifferenceFunction::getCenter(const Transformation &model) const {
    return this->box.getCenter() + model.translate;
}
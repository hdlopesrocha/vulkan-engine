#include "BoxDistanceFunction.hpp"

BoxDistanceFunction::BoxDistanceFunction() : SignedDistanceFunction(SdfType::BOX) {}

float BoxDistanceFunction::distance(const glm::vec3 &p, const Transformation &model) {
    glm::vec3 pos = p - getCenter(model); // Move point into model space
    pos = glm::inverse(model.quaternion) * pos;
    return SDF::box(pos, model.scale);
}



#include "BoxDistanceFunction.hpp"

BoxDistanceFunction::BoxDistanceFunction() {

}

float BoxDistanceFunction::distance(const glm::vec3 &p, const Transformation &model) {
    glm::vec3 pos = p - getCenter(model); // Move point into model space
    pos = glm::inverse(model.quaternion) * pos;
    return SDF::box(pos, model.scale);
}

SdfType BoxDistanceFunction::getType() const {
    return SdfType::BOX;
}

glm::vec3 BoxDistanceFunction::getCenter(const Transformation &model) const {
    return model.translate;
}



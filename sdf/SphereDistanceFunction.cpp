#include "SphereDistanceFunction.hpp"


SphereDistanceFunction::SphereDistanceFunction() {
    
}

float SphereDistanceFunction::distance(const glm::vec3 &p, const Transformation &model) {
    glm::vec3 pos = p - model.translate;
    pos = glm::inverse(model.quaternion) * pos;

    glm::vec3 radii = model.scale;
    glm::vec3 q = glm::abs(pos) / radii;
    return (glm::length(q) - 1.0f) * glm::min(glm::min(radii.x, radii.y), radii.z);
}

SdfType SphereDistanceFunction::getType() const {
    return SdfType::SPHERE;
}

glm::vec3 SphereDistanceFunction::getCenter(const Transformation &model) const {
    return model.translate;
}



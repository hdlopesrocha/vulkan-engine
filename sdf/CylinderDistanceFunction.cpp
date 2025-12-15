#include "CylinderDistanceFunction.hpp"
#include "SDF.hpp"


CylinderDistanceFunction::CylinderDistanceFunction() {
    
}

float CylinderDistanceFunction::distance(const glm::vec3 &p, const Transformation &model) {
 // Transform point into local space
    glm::vec3 pos = p - getCenter(model);
    pos = glm::inverse(model.quaternion) * pos;
    glm::vec3 q = pos / model.scale;

    // Distance in local space
    float d = SDF::cylinder(q, 0.5f, 1.0f);

    // Rescale back to world space
    float minScale = glm::min(glm::min(model.scale.x, model.scale.y), model.scale.z);
    return d * minScale;
}

SdfType CylinderDistanceFunction::getType() const {
    return SdfType::CYLINDER;
}

glm::vec3 CylinderDistanceFunction::getCenter(const Transformation &model) const {
    return model.translate;
}


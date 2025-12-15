#include "ConeDistanceFunction.hpp"
#include "SDF.hpp"


ConeDistanceFunction::ConeDistanceFunction() {
    
}

float ConeDistanceFunction::distance(const glm::vec3 &p, const Transformation &model) {
 // Transform point into local space
    glm::vec3 pos = p - getCenter(model);
    pos = glm::inverse(model.quaternion) * pos;
    glm::vec3 q = pos / model.scale - glm::vec3(0,1,0);

    // Distance in local space
    float d = SDF::cone(q);

    // Rescale back to world space
    float minScale = glm::min(glm::min(model.scale.x, model.scale.y), model.scale.z);
    return d * minScale;
}

SdfType ConeDistanceFunction::getType() const {
    return SdfType::CONE;
}

glm::vec3 ConeDistanceFunction::getCenter(const Transformation &model) const {
    return model.translate;
}



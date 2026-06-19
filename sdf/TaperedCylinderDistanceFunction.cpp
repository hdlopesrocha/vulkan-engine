#include "TaperedCylinderDistanceFunction.hpp"
#include "SDF.hpp"

TaperedCylinderDistanceFunction::TaperedCylinderDistanceFunction(float r1, float r2)
    : r1(r1), r2(r2) {
}

float TaperedCylinderDistanceFunction::distance(const glm::vec3 &p, const Transformation &model) {
    // Transform point into local space
    glm::vec3 pos = p - getCenter(model);
    pos = glm::inverse(model.quaternion) * pos;
    glm::vec3 q = pos / model.scale;

    // Distance in local space (unit cylinder: h=0.5, r1/r2 are in unit space)
    float d = SDF::taperedCylinder(q, r1, r2, 0.5f);

    // Rescale back to world space
    float minScale = glm::min(glm::min(model.scale.x, model.scale.y), model.scale.z);
    return d * minScale;
}

SdfType TaperedCylinderDistanceFunction::getType() const {
    return SdfType::TAPERED_CYLINDER;
}

glm::vec3 TaperedCylinderDistanceFunction::getCenter(const Transformation &model) const {
    return model.translate;
}

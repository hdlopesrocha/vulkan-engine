#include "TaperedCapsuleDistanceFunction.hpp"
#include "SDF.hpp"

TaperedCapsuleDistanceFunction::TaperedCapsuleDistanceFunction(glm::vec3 a, glm::vec3 b, float r1, float r2)
    : SignedDistanceFunction(SdfType::TAPERED_CAPSULE), a(a), b(b), r1(r1), r2(r2) {
}

float TaperedCapsuleDistanceFunction::distance(const glm::vec3 &p, const Transformation &model) {
    glm::vec3 pos = p - model.translate;
    pos = glm::inverse(model.quaternion) * pos;
    return SDF::taperedCapsule(pos / model.scale, a, b, r1, r2);
}

glm::vec3 TaperedCapsuleDistanceFunction::getCenter(const Transformation &model) const {
    return 0.5f * (a + b) + model.translate;
}

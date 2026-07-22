#include "TaperedCapsuleDistanceFunction.hpp"
#include "SDF.hpp"

TaperedCapsuleDistanceFunction::TaperedCapsuleDistanceFunction(glm::vec3 a_, glm::vec3 b_, float r1_, float r2_)
    : SignedDistanceFunction(SdfType::TAPERED_CAPSULE), a(a_), b(b_), r1(r1_), r2(r2_) {
}

float TaperedCapsuleDistanceFunction::distance(const glm::vec3 &p, const Transformation &model) {
    glm::vec3 pos = p - model.translate;
    pos = glm::inverse(model.quaternion) * pos;
    float d = SDF::taperedCapsule(pos / model.scale, a, b, r1, r2);
    float minScale = glm::min(glm::min(model.scale.x, model.scale.y), model.scale.z);
    return d * minScale;
}

glm::vec3 TaperedCapsuleDistanceFunction::getCenter(const Transformation &model) const {
    return 0.5f * (a + b) + model.translate;
}

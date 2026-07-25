#include "TaperedCylinderDistanceFunction.hpp"
#include "SDF.hpp"

TaperedCylinderDistanceFunction::TaperedCylinderDistanceFunction(float r1_, float r2_,
                                                                const Transformation &model, float bias)
    : SignedDistanceFunction(SdfType::TAPERED_CYLINDER), r1(r1_), r2(r2_)
    , sphere(getSphere(model, bias)) {}

float TaperedCylinderDistanceFunction::distance(const glm::vec3 &p, const Transformation &model) {
    glm::vec3 pos = p - getCenter(model);
    pos = glm::inverse(model.quaternion) * pos;
    glm::vec3 q = pos / model.scale;

    float d = SDF::taperedCylinder(q, r1, r2, 0.5f);

    float minScale = glm::min(glm::min(model.scale.x, model.scale.y), model.scale.z);
    return d * minScale;
}

BoundingSphere TaperedCylinderDistanceFunction::getSphere(const Transformation &model, float bias) const {
    float maxRadius = glm::max(r1, r2);
    return BoundingSphere(getCenter(model), glm::length(model.scale) * glm::sqrt(maxRadius * maxRadius + 0.25f) + bias);
}

ContainmentType TaperedCylinderDistanceFunction::check(const BoundingCube &cube) const {
    return sphere.test(cube);
}

bool TaperedCylinderDistanceFunction::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
}

const char* TaperedCylinderDistanceFunction::getLabel() const {
    return "Tapered Cylinder";
}

#include "TaperedCylinderDistanceFunction.hpp"
#include "SDF.hpp"

TaperedCylinderDistanceFunction::TaperedCylinderDistanceFunction(float r1_, float r2_,
                                                                const Transformation &model, float bias)
    : SignedDistanceFunction(SdfType::TAPERED_CYLINDER, model.translate, model), r1(r1_), r2(r2_)
    , sphere(getSphere(model, bias)) {}

float TaperedCylinderDistanceFunction::distance(const glm::vec3 &p) const {
    glm::vec3 pos = p - getCenter();
    pos = glm::inverse(m_model.quaternion) * pos;
    glm::vec3 q = pos / m_model.scale;

    float d = SDF::taperedCylinder(q, r1, r2, 0.5f);

    float minScale = glm::min(glm::min(m_model.scale.x, m_model.scale.y), m_model.scale.z);
    return d * minScale;
}

BoundingSphere TaperedCylinderDistanceFunction::getSphere(const Transformation &model, float bias) const {
    float maxRadius = glm::max(r1, r2);
    return BoundingSphere(getCenter(), glm::length(model.scale) * glm::sqrt(maxRadius * maxRadius + 0.25f) + bias);
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

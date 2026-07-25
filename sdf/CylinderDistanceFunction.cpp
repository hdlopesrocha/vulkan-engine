#include "CylinderDistanceFunction.hpp"
#include "SDF.hpp"

CylinderDistanceFunction::CylinderDistanceFunction(const Transformation &model, float bias)
    : SignedDistanceFunction(SdfType::CYLINDER)
    , sphere(getSphere(model, bias)) {}

float CylinderDistanceFunction::distance(const glm::vec3 &p, const Transformation &model) {
    glm::vec3 pos = p - getCenter(model);
    pos = glm::inverse(model.quaternion) * pos;
    glm::vec3 q = pos / model.scale;

    float d = SDF::cylinder(q, 0.5f, 1.0f);

    float minScale = glm::min(glm::min(model.scale.x, model.scale.y), model.scale.z);
    return d * minScale;
}

BoundingSphere CylinderDistanceFunction::getSphere(const Transformation &model, float bias) const {
    return BoundingSphere(getCenter(model), glm::length(model.scale)*sqrt(0.5f) + bias);
}

ContainmentType CylinderDistanceFunction::check(const BoundingCube &cube) const {
    return sphere.test(cube);
}

bool CylinderDistanceFunction::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
}

const char* CylinderDistanceFunction::getLabel() const {
    return "Cylinder";
}

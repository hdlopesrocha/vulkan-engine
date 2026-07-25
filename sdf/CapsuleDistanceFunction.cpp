#include "CapsuleDistanceFunction.hpp"

CapsuleDistanceFunction::CapsuleDistanceFunction(glm::vec3 a_, glm::vec3 b_, float r, const Transformation &model, float bias)
    : SignedDistanceFunction(SdfType::CAPSULE, 0.5f*(a_+b_)+model.translate, model), a(a_), b(b_), radius(r), sphere(getSphere(model, bias)) {}

float CapsuleDistanceFunction::distance(const glm::vec3 &p) const {
    glm::vec3 pos = p - m_model.translate;
    pos = glm::inverse(m_model.quaternion) * pos;
    float d = SDF::capsule(pos / m_model.scale, a, b, radius);
    float minScale = glm::min(glm::min(m_model.scale.x, m_model.scale.y), m_model.scale.z);
    return d * minScale;
}

BoundingSphere CapsuleDistanceFunction::getSphere(const Transformation &model, float bias) const {
    float halfLen = glm::length(b - a) * 0.5f;
    float maxR = radius;
    glm::vec3 center = model.translate + model.quaternion * (model.scale * 0.5f * (a + b));
    return BoundingSphere(center, glm::length(model.scale) * (halfLen + maxR) + bias);
}

ContainmentType CapsuleDistanceFunction::check(const BoundingCube &cube) const {
    return sphere.test(cube);
}

bool CapsuleDistanceFunction::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
}

const char* CapsuleDistanceFunction::getLabel() const {
    return "Capsule";
}

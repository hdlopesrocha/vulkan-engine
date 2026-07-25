#include "OctahedronDistanceFunction.hpp"

OctahedronDistanceFunction::OctahedronDistanceFunction(const Transformation &model, float bias)
    : SignedDistanceFunction(SdfType::OCTAHEDRON, model.translate, model)
    , sphere(getSphere(model, bias)) {}

float OctahedronDistanceFunction::distance(const glm::vec3 &p) const {
    glm::vec3 pos = p - getCenter();
    pos = glm::inverse(m_model.quaternion) * pos;

    glm::vec3 q = pos / m_model.scale;

    float d = SDF::octahedron(q, 1.0f);
    float minScale = glm::min(glm::min(m_model.scale.x, m_model.scale.y), m_model.scale.z);

    return d * minScale;
}

BoundingSphere OctahedronDistanceFunction::getSphere(const Transformation &model, float bias) const {
    return BoundingSphere(getCenter(), glm::length(model.scale)*sqrt(0.5f) + bias);
}

ContainmentType OctahedronDistanceFunction::check(const BoundingCube &cube) const {
    return sphere.test(cube);
}

bool OctahedronDistanceFunction::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
}

const char* OctahedronDistanceFunction::getLabel() const {
    return "Octahedron";
}

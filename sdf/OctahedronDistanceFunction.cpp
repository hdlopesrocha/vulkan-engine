#include "OctahedronDistanceFunction.hpp"

OctahedronDistanceFunction::OctahedronDistanceFunction(const Transformation &model, float bias)
    : SignedDistanceFunction(SdfType::OCTAHEDRON)
    , sphere(getSphere(model, bias)) {}

float OctahedronDistanceFunction::distance(const glm::vec3 &p, const Transformation &model) const {
    glm::vec3 pos = p - getCenter(model);
    pos = glm::inverse(model.quaternion) * pos;

    glm::vec3 q = pos / model.scale;

    float d = SDF::octahedron(q, 1.0f);
    float minScale = glm::min(glm::min(model.scale.x, model.scale.y), model.scale.z);

    return d * minScale;
}

BoundingSphere OctahedronDistanceFunction::getSphere(const Transformation &model, float bias) const {
    return BoundingSphere(getCenter(model), glm::length(model.scale)*sqrt(0.5f) + bias);
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

#include "PyramidDistanceFunction.hpp"

PyramidDistanceFunction::PyramidDistanceFunction(const Transformation &model, float bias)
    : SignedDistanceFunction(SdfType::PYRAMID, model.translate)
    , sphere(getSphere(model, bias)) {}

float PyramidDistanceFunction::distance(const glm::vec3 &p, const Transformation &model) const {
   glm::vec3 pos = p - getCenter();
    pos = glm::inverse(model.quaternion) * pos;

    pos /= model.scale;

    float d = SDF::pyramid(pos, 1.0f, sqrt(0.5f));

    float minScale = glm::min(glm::min(model.scale.x, model.scale.y), model.scale.z);
    return d * minScale;
}

float PyramidDistanceFunction::boundingSphereRadius(float width, float depth, float height) const {
    return glm::length(glm::vec3(width, height, depth));
}

BoundingSphere PyramidDistanceFunction::getSphere(const Transformation &model, float bias) const {
    return BoundingSphere(getCenter(), sqrt(0.5f) * glm::length(model.scale) + bias);
}

ContainmentType PyramidDistanceFunction::check(const BoundingCube &cube) const {
    return sphere.test(cube);
}

bool PyramidDistanceFunction::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
}

const char* PyramidDistanceFunction::getLabel() const {
    return "Pyramid";
}

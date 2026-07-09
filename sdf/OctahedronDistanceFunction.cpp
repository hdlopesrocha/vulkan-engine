#include "OctahedronDistanceFunction.hpp"


OctahedronDistanceFunction::OctahedronDistanceFunction() : SignedDistanceFunction(SdfType::OCTAHEDRON) {}

float OctahedronDistanceFunction::distance(const glm::vec3 &p, const Transformation &model) {
    glm::vec3 pos = p - getCenter(model);
    pos = glm::inverse(model.quaternion) * pos;

    glm::vec3 q = pos / model.scale;

    float d = SDF::octahedron(q, 1.0f);
    float minScale = glm::min(glm::min(model.scale.x, model.scale.y), model.scale.z);

    return d * minScale;
}


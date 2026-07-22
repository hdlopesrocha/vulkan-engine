#include "TorusDistanceFunction.hpp"

TorusDistanceFunction::TorusDistanceFunction(glm::vec2 radius_) : SignedDistanceFunction(SdfType::TORUS), radius(radius_) {}

float TorusDistanceFunction::distance(const glm::vec3 &p, const Transformation &model) {
     glm::vec3 pos = p - getCenter(model);
    pos = glm::inverse(model.quaternion) * pos;

    glm::vec3 q = pos / model.scale;
    float d = SDF::torus(q, radius);

    float minScale = glm::min(glm::min(model.scale.x, model.scale.y), model.scale.z);
    return d * minScale;
}


#include "TriangleStripDistanceFunction.hpp"
#include "SdfType.hpp"

TriangleStripDistanceFunction::TriangleStripDistanceFunction(
    const glm::vec3& v0, const glm::vec3& v1,
    const glm::vec3& v2, const glm::vec3& v3, float halfThick)
    : SignedDistanceFunction(SdfType::TRIANGLE_STRIP), v0(v0), v1(v1), v2(v2), v3(v3), halfThick(halfThick)
{
}

float TriangleStripDistanceFunction::distance(const glm::vec3 &p, const Transformation &model) {
    glm::vec3 pos = p - getCenter(model);
    pos = glm::inverse(model.quaternion) * pos;
    glm::vec3 q = pos / model.scale;

    float d = SDF::triangleStrip(q, v0, v1, v2, v3, halfThick);

    float minScale = glm::min(glm::min(model.scale.x, model.scale.y), model.scale.z);
    return d * minScale;
}

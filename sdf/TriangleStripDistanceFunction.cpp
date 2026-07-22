#include "TriangleStripDistanceFunction.hpp"
#include "SdfType.hpp"

TriangleStripDistanceFunction::TriangleStripDistanceFunction(
    const glm::vec3& v0_, const glm::vec3& v1_,
    const glm::vec3& v2_, const glm::vec3& v3_, float halfThick_)
    : SignedDistanceFunction(SdfType::TRIANGLE_STRIP), v0(v0_), v1(v1_), v2(v2_), v3(v3_), halfThick(halfThick_)
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

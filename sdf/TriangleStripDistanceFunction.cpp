#include "TriangleStripDistanceFunction.hpp"
#include "SdfType.hpp"
#include <algorithm>

TriangleStripDistanceFunction::TriangleStripDistanceFunction(
    const glm::vec3& v0_, const glm::vec3& v1_,
    const glm::vec3& v2_, const glm::vec3& v3_, float halfThick_,
    const glm::vec3& sphereCenter, float sphereRadius,
    const Transformation &model, float bias)
    : SignedDistanceFunction(SdfType::TRIANGLE_STRIP, sphereCenter, model)
    , v0(v0_), v1(v1_), v2(v2_), v3(v3_), halfThick(halfThick_)
    , m_sphereCenter(sphereCenter)
    , m_sphereRadius(sphereRadius)
    , sphere(getSphere(model, bias))
{
}

float TriangleStripDistanceFunction::distance(const glm::vec3 &p) const {
    glm::vec3 pos = p - m_model.translate;
    pos = glm::inverse(m_model.quaternion) * pos;
    glm::vec3 q = pos / m_model.scale;

    float stripDist = SDF::triangleStrip(q, v0, v1, v2, v3, halfThick);

    float minScale = glm::min(glm::min(m_model.scale.x, m_model.scale.y), m_model.scale.z);
    float d = stripDist * minScale;

    glm::vec3 dp = p - m_sphereCenter;
    float sphereDist = glm::length(dp) - m_sphereRadius;
    return std::max(d, sphereDist);
}

BoundingSphere TriangleStripDistanceFunction::getSphere(const Transformation &model, float bias) const {
    return BoundingSphere(m_sphereCenter, m_sphereRadius + bias);
}

ContainmentType TriangleStripDistanceFunction::check(const BoundingCube &cube) const {
    return sphere.test(cube);
}

bool TriangleStripDistanceFunction::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
}

const char* TriangleStripDistanceFunction::getLabel() const {
    return "TriangleStrip";
}

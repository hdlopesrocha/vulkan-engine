#include "WrappedTriangleStrip.hpp"
#include <glm/glm.hpp>
#include <algorithm>

WrappedTriangleStrip::WrappedTriangleStrip(TriangleStripDistanceFunction* function,
                                           const glm::vec3& sphereCenter, float sphereRadius)
    : WrappedSignedDistanceFunction(function)
    , m_sphereCenter(sphereCenter)
    , m_sphereRadius(sphereRadius)
{
}

WrappedTriangleStrip::~WrappedTriangleStrip() {
}

float WrappedTriangleStrip::distance(const glm::vec3 &p, const Transformation &model) {
    float stripDist = function->distance(p, model);
    glm::vec3 d = p - m_sphereCenter;
    float sphereDist = glm::length(d) - m_sphereRadius;
    return std::max(stripDist, sphereDist);
}

BoundingSphere WrappedTriangleStrip::getSphere(const Transformation &model, float bias) const {
    return BoundingSphere(m_sphereCenter, m_sphereRadius + bias);
}

ContainmentType WrappedTriangleStrip::check(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingSphere sphere = getSphere(model, bias);
    return sphere.test(cube);
}

bool WrappedTriangleStrip::isContained(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingSphere sphere = getSphere(model, bias);
    return cube.contains(sphere);
}

float WrappedTriangleStrip::getLength(const Transformation &model, float bias) const {
    return m_sphereRadius * 2.0f + bias;
}

void WrappedTriangleStrip::accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const {
    getSphere(model, bias).accept(visitor);
}

const char* WrappedTriangleStrip::getLabel() const {
    return "TriangleStrip";
}

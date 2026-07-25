#include "WrappedTriangleStrip.hpp"
#include <glm/glm.hpp>
#include <algorithm>

WrappedTriangleStrip::WrappedTriangleStrip(TriangleStripDistanceFunction* function_,
                                           const glm::vec3& sphereCenter, float sphereRadius,
                                           const Transformation &model, float bias)
    : WrappedSignedDistanceFunction(function_)
    , m_sphereCenter(sphereCenter)
    , m_sphereRadius(sphereRadius)
    , sphere(getSphere(model, bias))
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

ContainmentType WrappedTriangleStrip::check(const BoundingCube &cube) const {
    return sphere.test(cube);
}

bool WrappedTriangleStrip::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
}

const char* WrappedTriangleStrip::getLabel() const {
    return "TriangleStrip";
}

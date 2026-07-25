#include "WrappedRoad.hpp"
#include <glm/glm.hpp>
#include <algorithm>

WrappedRoad::WrappedRoad(RoadDistanceFunction* function_,
                         const glm::vec3& sphereCenter, float sphereRadius,
                         const Transformation &model, float bias)
    : WrappedSignedDistanceFunction(function_)
    , m_sphereCenter(sphereCenter)
    , m_sphereRadius(sphereRadius)
    , sphere(getSphere(model, bias))
{
}

WrappedRoad::~WrappedRoad() {
}

float WrappedRoad::distance(const glm::vec3 &p, const Transformation &model) {
    float roadDist = function->distance(p, model);
    glm::vec3 d = p - m_sphereCenter;
    float sphereDist = glm::length(d) - m_sphereRadius;
    return std::max(roadDist, sphereDist);
}

BoundingSphere WrappedRoad::getSphere(const Transformation &model, float bias) const {
    return BoundingSphere(m_sphereCenter, m_sphereRadius + bias);
}

ContainmentType WrappedRoad::check(const BoundingCube &cube) const {
    return sphere.test(cube);
}

bool WrappedRoad::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
}


const char* WrappedRoad::getLabel() const {
    return "Road";
}

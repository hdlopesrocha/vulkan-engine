#include "WrappedRoad.hpp"
#include <glm/glm.hpp>
#include <algorithm>

WrappedRoad::WrappedRoad(RoadDistanceFunction* function,
                         const glm::vec3& sphereCenter, float sphereRadius)
    : WrappedSignedDistanceFunction(function)
    , m_sphereCenter(sphereCenter)
    , m_sphereRadius(sphereRadius)
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

ContainmentType WrappedRoad::check(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingSphere sphere = getSphere(model, bias);
    return sphere.test(cube);
}

bool WrappedRoad::isContained(const BoundingCube &cube, const Transformation &model, float bias) const {
    BoundingSphere sphere = getSphere(model, bias);
    return cube.contains(sphere);
}

float WrappedRoad::getLength(const Transformation &model, float bias) const {
    RoadDistanceFunction* f = (RoadDistanceFunction*)function;
    return (f->getTMax() - f->getTMin()) * f->getSpline()->totalLength() + bias;
}

void WrappedRoad::accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const {
    getSphere(model, bias).accept(visitor);
}

const char* WrappedRoad::getLabel() const {
    return "Road";
}

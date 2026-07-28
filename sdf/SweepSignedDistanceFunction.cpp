#include "SweepSignedDistanceFunction.hpp"
#include "SphereDistanceFunction.hpp"
#include "BoxDistanceFunction.hpp"
#include "CapsuleDistanceFunction.hpp"
#include "OctahedronDistanceFunction.hpp"
#include "PyramidDistanceFunction.hpp"
#include "TorusDistanceFunction.hpp"
#include "ConeDistanceFunction.hpp"
#include "CylinderDistanceFunction.hpp"
#include "TaperedCylinderDistanceFunction.hpp"
#include "TaperedCapsuleDistanceFunction.hpp"

template<typename T>
SweepSignedDistanceFunction<T>::SweepSignedDistanceFunction(const T &f1, const T &f2,
                                                            const Transformation &model, float bias)
    : SignedDistanceFunction(SdfType::SWEEP, model)
    , function1(f1)
    , function2(f2)
    , posA(f1.getCenter())
    , posB(f2.getCenter())
{
    m_center = (posA + posB) * 0.5f;
    BoundingSphere s1 = function1.getSphere(model, bias);
    BoundingSphere s2 = function2.getSphere(model, bias);
    glm::vec3 delta = s2.center - s1.center;
    float dist = glm::length(delta);
    if (dist < 1e-6f) {
        sphere = BoundingSphere(s1.center, glm::max(s1.radius, s2.radius));
    } else {
        float r = (dist + s1.radius + s2.radius) * 0.5f;
        glm::vec3 c = s1.center + delta * ((r - s1.radius) / dist);
        sphere = BoundingSphere(c, r);
    }
}

template<typename T>
float SweepSignedDistanceFunction<T>::distance(const glm::vec3 &p) const {
    glm::vec3 seg = posB - posA;
    float segLenSq = glm::dot(seg, seg);
    if (segLenSq < 1e-6f) {
        return function1.distance(p);
    }
    float t = glm::clamp(glm::dot(p - posA, seg) / segLenSq, 0.0f, 1.0f);
    glm::vec3 closest = posA + t * seg;
    glm::quat rotInterp = glm::slerp(function1.getRotation(), function2.getRotation(), t);
    glm::quat rotA = function1.getRotation();
    glm::vec3 scaleInterp = glm::mix(function1.getScale(), function2.getScale(), t);
    glm::vec3 scaleA = function1.getScale();
    glm::vec3 q = posA + rotA * ((scaleA / scaleInterp) * (glm::inverse(rotInterp) * (p - closest)));
    float minScaleInterp = glm::min(glm::min(scaleInterp.x, scaleInterp.y), scaleInterp.z);
    float minScaleA = glm::min(glm::min(scaleA.x, scaleA.y), scaleA.z);
    return function1.distance(q) * minScaleInterp / minScaleA;
}

template<typename T>
BoundingSphere SweepSignedDistanceFunction<T>::getSphere(const Transformation &model, float bias) const {
    return sphere;
}

template<typename T>
BoundingBox SweepSignedDistanceFunction<T>::getBox(float bias) const {
    BoundingSphere s1 = function1.getSphere(m_model, bias);
    BoundingSphere s2 = function2.getSphere(m_model, bias);
    glm::vec3 min1 = s1.center - glm::vec3(s1.radius);
    glm::vec3 max1 = s1.center + glm::vec3(s1.radius);
    glm::vec3 min2 = s2.center - glm::vec3(s2.radius);
    glm::vec3 max2 = s2.center + glm::vec3(s2.radius);
    return BoundingBox(glm::min(min1, min2), glm::max(max1, max2));
}

template<typename T>
ContainmentType SweepSignedDistanceFunction<T>::check(const BoundingCube &cube) const {
    return sphere.test(cube);
}

template<typename T>
bool SweepSignedDistanceFunction<T>::isContained(const BoundingCube &cube) const {
    return cube.contains(sphere);
}

template<typename T>
const char* SweepSignedDistanceFunction<T>::getLabel() const {
    return "Sweep";
}

template<typename T>
glm::vec3 SweepSignedDistanceFunction<T>::getCenter() const {
    return m_center;
}

template class SweepSignedDistanceFunction<SphereDistanceFunction>;
template class SweepSignedDistanceFunction<BoxDistanceFunction>;
template class SweepSignedDistanceFunction<CapsuleDistanceFunction>;
template class SweepSignedDistanceFunction<OctahedronDistanceFunction>;
template class SweepSignedDistanceFunction<PyramidDistanceFunction>;
template class SweepSignedDistanceFunction<TorusDistanceFunction>;
template class SweepSignedDistanceFunction<ConeDistanceFunction>;
template class SweepSignedDistanceFunction<CylinderDistanceFunction>;
template class SweepSignedDistanceFunction<TaperedCylinderDistanceFunction>;
template class SweepSignedDistanceFunction<TaperedCapsuleDistanceFunction>;

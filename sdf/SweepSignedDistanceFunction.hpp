#pragma once
#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>
#include "../math/Transformation.hpp"
#include "../math/BoundingSphere.hpp"
#include "../math/BoundingBox.hpp"

template<typename T>
class SweepSignedDistanceFunction : public SignedDistanceFunction {
    T function1;
    T function2;
    glm::vec3 posA;
    glm::vec3 posB;
    BoundingSphere sphere;
public:
    SweepSignedDistanceFunction(const T &f1, const T &f2,
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

    virtual ~SweepSignedDistanceFunction() = default;

    float distance(const glm::vec3 &p) const override {
        float d1 = function1.distance(p);
        float d2 = function2.distance(p);
        glm::vec3 seg = posB - posA;
        float segLenSq = glm::dot(seg, seg);
        if (segLenSq < 1e-6f) {
            return d1;
        }
        float t = glm::clamp(glm::dot(p - posA, seg) / segLenSq, 0.0f, 1.0f);
        glm::vec3 closest = posA + t * seg;
        float dSwept = function2.distance(p - closest + posB);
        return glm::min(glm::min(d1, d2), dSwept);
    }

    BoundingSphere getSphere(const Transformation &model, float bias) const override {
        return sphere;
    }

    BoundingBox getBox(float bias) const override {
        BoundingSphere s1 = function1.getSphere(m_model, bias);
        BoundingSphere s2 = function2.getSphere(m_model, bias);
        glm::vec3 min1 = s1.center - glm::vec3(s1.radius);
        glm::vec3 max1 = s1.center + glm::vec3(s1.radius);
        glm::vec3 min2 = s2.center - glm::vec3(s2.radius);
        glm::vec3 max2 = s2.center + glm::vec3(s2.radius);
        return BoundingBox(glm::min(min1, min2), glm::max(max1, max2));
    }

    ContainmentType check(const BoundingCube &cube) const override {
        return sphere.test(cube);
    }

    bool isContained(const BoundingCube &cube) const override {
        return cube.contains(sphere);
    }

    const char* getLabel() const override {
        return "Sweep";
    }

    glm::vec3 getCenter() const override {
        return m_center;
    }
};

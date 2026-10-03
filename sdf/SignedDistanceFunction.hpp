#pragma once
#include <glm/glm.hpp>
#include "../math/Transformation.hpp"
#include "../math/BoundingVolume.hpp"
#include "../math/BoundingCube.hpp"
#include "../math/BoundingSphere.hpp"
#include "../math/BoundingBox.hpp"

class SignedDistanceFunction {
protected:
    // Bounding sphere used by the shared check()/isContained() implementations.
    // Derived primitives fill it in their constructor body (getSphere() must be
    // called after the base is constructed so it can read getCenter()).
    BoundingSphere m_sphere;
    glm::vec3 m_center{};
    Transformation m_model{};
    SignedDistanceFunction() {}
    SignedDistanceFunction(const Transformation &model) : m_model(model) {}
    SignedDistanceFunction(const glm::vec3 &center, const Transformation &model) : m_center(center), m_model(model) {}

    // Shared isotropic bounding sphere (radius = |scale| * sqrt(0.5) + bias).
    // Deliberately a helper instead of the default getSphere() body: subclasses
    // that do not override getSphere() must keep returning an empty sphere.
    static BoundingSphere isotropicSphere(const glm::vec3 &center, const Transformation &model, float bias) {
        return BoundingSphere(center, glm::length(model.scale) * glm::sqrt(0.5f) + bias);
    }
public:
    virtual ~SignedDistanceFunction() = default;
    virtual float distance(const glm::vec3 &p) const = 0;
    virtual glm::vec3 getCenter() const { return m_center; }
    glm::quat getRotation() const { return m_model.quaternion; }
    glm::vec3 getScale() const { return m_model.scale; }
    glm::vec3 getPosition() const { return m_model.translate; }

    virtual ContainmentType check(const BoundingCube &cube) const { return m_sphere.test(cube); }
    virtual bool isContained(const BoundingCube &cube) const { return cube.contains(m_sphere); }
    virtual BoundingSphere getSphere(const Transformation &model, float bias) const { return BoundingSphere(); }
    virtual BoundingBox getBox(float bias) const { return BoundingBox(); }
};

 
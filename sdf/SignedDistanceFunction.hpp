#pragma once
#include <glm/glm.hpp>
#include "../math/Transformation.hpp"
#include "../math/BoundingVolume.hpp"
#include "../math/BoundingCube.hpp"
#include "../math/BoundingSphere.hpp"
#include "../math/BoundingBox.hpp"
#include "SdfType.hpp"

class SignedDistanceFunction {
protected:
    SdfType type;
    glm::vec3 m_center{};
    Transformation m_model{};
    SignedDistanceFunction() : type() {}
    SignedDistanceFunction(const Transformation &model) : type(), m_model(model) {}
    SignedDistanceFunction(SdfType t, const Transformation &model) : type(t), m_model(model) {}
    SignedDistanceFunction(SdfType t, const glm::vec3 &center, const Transformation &model) : type(t), m_center(center), m_model(model) {}
public:
    virtual ~SignedDistanceFunction() = default;
    virtual float distance(const glm::vec3 &p) const = 0;
    virtual glm::vec3 getCenter() const { return m_center; }
    glm::quat getRotation() const { return m_model.quaternion; }
    glm::vec3 getScale() const { return m_model.scale; }
    glm::vec3 getPosition() const { return m_model.translate; }
    virtual const char* getLabel() const { return ""; }
    virtual SdfType getType() const { return type; }

    virtual ContainmentType check(const BoundingCube &cube) const { return ContainmentType::Intersects; }
    virtual bool isContained(const BoundingCube &cube) const { return false; }
    virtual BoundingSphere getSphere(const Transformation &model, float bias) const { return BoundingSphere(); }
    virtual BoundingBox getBox(float bias) const { return BoundingBox(); }
};

 
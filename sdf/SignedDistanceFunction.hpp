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
    SignedDistanceFunction() : type() {}
    SignedDistanceFunction(SdfType t) : type(t) {}
public:
    virtual ~SignedDistanceFunction() = default;
    virtual float distance(const glm::vec3 &p, const Transformation &model) const = 0;
    virtual glm::vec3 getCenter(const Transformation &model) const { return model.translate; }
    virtual const char* getLabel() const { return ""; }
    virtual SdfType getType() const { return type; }

    virtual ContainmentType check(const BoundingCube &cube) const { return ContainmentType::Intersects; }
    virtual bool isContained(const BoundingCube &cube) const { return false; }
    virtual BoundingSphere getSphere(const Transformation &model, float bias) const { return BoundingSphere(); }
    virtual BoundingBox getBox(float bias) const { return BoundingBox(); }
};

 
#pragma once
#include <glm/glm.hpp>
#include "../math/Transformation.hpp"
#include "SdfType.hpp"

class SignedDistanceFunction {
protected:
    SdfType type;
    SignedDistanceFunction() : type() {}
    SignedDistanceFunction(SdfType t) : type(t) {}
public:
    virtual ~SignedDistanceFunction() = default;
    virtual float distance(const glm::vec3 &p, const Transformation &model) = 0;
    virtual glm::vec3 getCenter(const Transformation &model) const { return model.translate; }
    virtual const char* getLabel() const { return ""; }
    virtual SdfType getType() const { return type; }
};

 
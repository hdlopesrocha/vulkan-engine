#pragma once
#include <glm/glm.hpp>
#include "../math/Transformation.hpp"
#include "SdfType.hpp"

class SignedDistanceFunction {
public:
    virtual ~SignedDistanceFunction() = default;
    virtual float distance(const glm::vec3 &p, const Transformation &model) = 0;
    virtual glm::vec3 getCenter(const Transformation &model) const = 0;
    // Provide a default label so simple distance functions need not override it.
    virtual const char* getLabel() const { return ""; }
    virtual SdfType getType() const = 0;
};

 
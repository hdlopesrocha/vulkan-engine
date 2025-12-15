#pragma once
#include <glm/glm.hpp>
#include "../math/Transformation.hpp"
#include "SdfType.hpp"

class SignedDistanceFunction {
public:
    virtual ~SignedDistanceFunction() = default;
    virtual float distance(const glm::vec3 &p, const Transformation &model) = 0;
    virtual glm::vec3 getCenter(const Transformation &model) const = 0;
    virtual const char* getLabel() const = 0;
    virtual SdfType getType() const = 0;
};

 
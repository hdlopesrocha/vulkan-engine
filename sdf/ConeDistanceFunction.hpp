#pragma once
#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>
#include "../math/BoundingSphere.hpp"

class ConeDistanceFunction : public SignedDistanceFunction {
public:
    ConeDistanceFunction(const Transformation &model, float bias);
    virtual ~ConeDistanceFunction() = default;
    float distance(const glm::vec3 &p) const override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
};

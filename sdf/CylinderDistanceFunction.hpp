#pragma once
#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>
#include "../math/BoundingSphere.hpp"

class CylinderDistanceFunction : public SignedDistanceFunction {
public:
    CylinderDistanceFunction(const Transformation &model, float bias);
    virtual ~CylinderDistanceFunction() = default;
    float distance(const glm::vec3 &p) const override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
};

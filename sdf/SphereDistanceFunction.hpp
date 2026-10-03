#pragma once
#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>
#include "../math/BoundingSphere.hpp"

class SphereDistanceFunction : public SignedDistanceFunction {
public:
    SphereDistanceFunction(const Transformation &model, float bias);
    virtual ~SphereDistanceFunction() = default;
    float distance(const glm::vec3 &p) const override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
};

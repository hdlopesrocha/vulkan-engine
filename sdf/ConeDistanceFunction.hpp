#pragma once
#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>
#include "../math/BoundingSphere.hpp"

class ConeDistanceFunction : public SignedDistanceFunction {
private:
    BoundingSphere sphere;
public:
    ConeDistanceFunction(const Transformation &model, float bias);
    virtual ~ConeDistanceFunction() = default;
    float distance(const glm::vec3 &p, const Transformation &model) override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    const char* getLabel() const override;
};

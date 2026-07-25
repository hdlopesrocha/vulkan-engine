#pragma once
#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>
#include "../math/BoundingSphere.hpp"

class CylinderDistanceFunction : public SignedDistanceFunction {
private:
    BoundingSphere sphere;
public:
    CylinderDistanceFunction(const Transformation &model, float bias);
    virtual ~CylinderDistanceFunction() = default;
    float distance(const glm::vec3 &p) const override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    const char* getLabel() const override;
};

#pragma once
#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>
#include "../math/BoundingSphere.hpp"

class SphereDistanceFunction : public SignedDistanceFunction {
private:
    BoundingSphere sphere;
public:
    SphereDistanceFunction(const Transformation &model, float bias);
    virtual ~SphereDistanceFunction() = default;
    float distance(const glm::vec3 &p, const Transformation &model) override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    const char* getLabel() const override;
};

#pragma once
#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>
#include "../math/BoundingSphere.hpp"

class TaperedCylinderDistanceFunction : public SignedDistanceFunction {
public:
    float r1; // bottom radius (y = -h)
    float r2; // top radius    (y = +h)
private:
    BoundingSphere sphere;
public:
    TaperedCylinderDistanceFunction(float r1_ = 0.25f, float r2_ = 0.5f,
                                    const Transformation &model = Transformation(), float bias = 0.0f);
    virtual ~TaperedCylinderDistanceFunction() = default;
    float distance(const glm::vec3 &p) const override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    const char* getLabel() const override;
};

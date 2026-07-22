#pragma once
#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>

class TaperedCylinderDistanceFunction : public SignedDistanceFunction {
public:
    TaperedCylinderDistanceFunction(float r1_ = 0.25f, float r2_ = 0.5f);
    virtual ~TaperedCylinderDistanceFunction() = default;
    float distance(const glm::vec3 &p, const Transformation &model) override;

    float r1; // bottom radius (y = -h)
    float r2; // top radius    (y = +h)
};

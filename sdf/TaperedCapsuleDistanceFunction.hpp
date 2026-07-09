#pragma once
#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>

class TaperedCapsuleDistanceFunction : public SignedDistanceFunction {
public:
    glm::vec3 a;
    glm::vec3 b;
    float r1; // radius at point A
    float r2; // radius at point B

    TaperedCapsuleDistanceFunction(glm::vec3 a = glm::vec3(0.0f, -1.0f, 0.0f),
                                   glm::vec3 b = glm::vec3(0.0f, 1.0f, 0.0f),
                                   float r1 = 0.5f, float r2 = 0.25f);
    virtual ~TaperedCapsuleDistanceFunction() = default;
    float distance(const glm::vec3 &p, const Transformation &model) override;
    glm::vec3 getCenter(const Transformation &model) const override;
};

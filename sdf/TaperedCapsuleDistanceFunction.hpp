#pragma once
#include "SignedDistanceFunction.hpp"
#include <glm/glm.hpp>
#include "../math/BoundingSphere.hpp"

class TaperedCapsuleDistanceFunction : public SignedDistanceFunction {
public:
    glm::vec3 a;
    glm::vec3 b;
    float r1; // radius at point A
    float r2; // radius at point B
private:
    BoundingSphere sphere;
public:
    TaperedCapsuleDistanceFunction(glm::vec3 a_ = glm::vec3(0.0f, -1.0f, 0.0f),
                                   glm::vec3 b_ = glm::vec3(0.0f, 1.0f, 0.0f),
                                   float r1_ = 0.5f, float r2_ = 0.25f,
                                   const Transformation &model = Transformation(), float bias = 0.0f);
    virtual ~TaperedCapsuleDistanceFunction() = default;
    float distance(const glm::vec3 &p) const override;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    const char* getLabel() const override;
};

#pragma once
#include "SignedDistanceFunction.hpp"
#include "../math/BoundingSphere.hpp"
#include <glm/glm.hpp>

class SignedDistanceEffect : public SignedDistanceFunction {
    protected:
    SignedDistanceFunction &function;
    BoundingSphere sphere;
    public:
    SignedDistanceEffect(SignedDistanceFunction &function_, const Transformation &model, float bias);
    ~SignedDistanceEffect();
    void setFunction(SignedDistanceFunction &function_);
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    glm::vec3 getCenter(const Transformation &model) const override;
};

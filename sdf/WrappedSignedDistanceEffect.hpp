#pragma once
#include "WrappedSignedDistanceFunction.hpp"
#include <glm/glm.hpp>

class WrappedSignedDistanceEffect : public WrappedSignedDistanceFunction {
    protected:
    BoundingSphere sphere;
    public:
    WrappedSignedDistanceEffect(WrappedSignedDistanceFunction * function_, const Transformation &model = Transformation(), float bias = 0.0f);
    ~WrappedSignedDistanceEffect();
    void setFunction(WrappedSignedDistanceFunction * function_);
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
};

 

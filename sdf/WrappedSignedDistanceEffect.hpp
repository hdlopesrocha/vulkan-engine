#pragma once
#include "WrappedSignedDistanceFunction.hpp"
#include <glm/glm.hpp>

class WrappedSignedDistanceEffect : public WrappedSignedDistanceFunction {
    public:
    WrappedSignedDistanceEffect(WrappedSignedDistanceFunction * function);
    ~WrappedSignedDistanceEffect();
    void setFunction(WrappedSignedDistanceFunction * function);
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    bool isContained(const BoundingCube &cube, const Transformation &model, float bias) const override;
    float getLength(const Transformation &model, float bias) const override;
    void accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const override;
};

 

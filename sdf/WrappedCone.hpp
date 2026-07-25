#pragma once
#include "WrappedSignedDistanceFunction.hpp"
#include "ConeDistanceFunction.hpp"
#include "../math/BoundingSphere.hpp"

class WrappedCone : public WrappedSignedDistanceFunction {
private:
    BoundingSphere sphere;
public:
    WrappedCone(ConeDistanceFunction * function_, const Transformation &model = Transformation(), float bias = 0.0f);
    ~WrappedCone();
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    const char* getLabel() const override;
};

 

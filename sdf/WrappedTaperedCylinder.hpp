#pragma once
#include "WrappedSignedDistanceFunction.hpp"
#include "TaperedCylinderDistanceFunction.hpp"
#include "../math/BoundingSphere.hpp"

class WrappedTaperedCylinder : public WrappedSignedDistanceFunction {
private:
    BoundingSphere sphere;
public:
    WrappedTaperedCylinder(TaperedCylinderDistanceFunction * function_, const Transformation &model = Transformation(), float bias = 0.0f);
    ~WrappedTaperedCylinder();
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    const char* getLabel() const override;
};

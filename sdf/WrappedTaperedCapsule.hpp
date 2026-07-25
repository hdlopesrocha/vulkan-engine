#pragma once
#include "WrappedSignedDistanceFunction.hpp"
#include "TaperedCapsuleDistanceFunction.hpp"
#include "../math/BoundingBox.hpp"
#include "../math/BoundingSphere.hpp"

class WrappedTaperedCapsule : public WrappedSignedDistanceFunction {
private:
    BoundingSphere sphere;
public:
    WrappedTaperedCapsule(TaperedCapsuleDistanceFunction * function_, const Transformation &model = Transformation(), float bias = 0.0f);
    ~WrappedTaperedCapsule();
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    const char* getLabel() const override;
};

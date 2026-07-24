#pragma once
#include "WrappedSignedDistanceFunction.hpp"
#include "TaperedCapsuleDistanceFunction.hpp"
#include "../math/BoundingBox.hpp"
#include "../math/BoundingSphere.hpp"

class WrappedTaperedCapsule : public WrappedSignedDistanceFunction {
public:
    WrappedTaperedCapsule(TaperedCapsuleDistanceFunction * function_);
    ~WrappedTaperedCapsule();
    BoundingSphere getSphere(const Transformation &model, float bias) const;
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    bool isContained(const BoundingCube &cube, const Transformation &model, float bias) const override;
    void accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const override;
    const char* getLabel() const override;
};

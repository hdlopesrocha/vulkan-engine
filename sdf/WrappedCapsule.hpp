#pragma once
#include "WrappedSignedDistanceFunction.hpp"
#include "CapsuleDistanceFunction.hpp"
#include "../math/BoundingSphere.hpp"

class WrappedCapsule : public WrappedSignedDistanceFunction {
private:
    BoundingSphere sphere;
    public:
    WrappedCapsule(CapsuleDistanceFunction * function_, const Transformation &model, float bias);
    ~WrappedCapsule();
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    const char* getLabel() const override;
};

 

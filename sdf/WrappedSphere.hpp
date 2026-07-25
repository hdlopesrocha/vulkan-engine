#pragma once
#include "WrappedSignedDistanceFunction.hpp"
#include "SphereDistanceFunction.hpp"
#include "../math/BoundingSphere.hpp"

class WrappedSphere : public WrappedSignedDistanceFunction {
private:
    BoundingSphere sphere;
    public:
    WrappedSphere(SphereDistanceFunction * function_, const Transformation &model, float bias);
    ~WrappedSphere();
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    const char* getLabel() const override;
};

 

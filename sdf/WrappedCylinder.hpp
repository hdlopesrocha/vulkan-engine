#pragma once
#include "WrappedSignedDistanceFunction.hpp"
#include "CylinderDistanceFunction.hpp"
#include "../math/BoundingSphere.hpp"

class WrappedCylinder : public WrappedSignedDistanceFunction {
private:
    BoundingSphere sphere;
    public:
    WrappedCylinder(CylinderDistanceFunction * function_, const Transformation &model = Transformation(), float bias = 0.0f);
    ~WrappedCylinder();
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    const char* getLabel() const override;
};

 

#pragma once
#include "WrappedSignedDistanceFunction.hpp"
#include "TorusDistanceFunction.hpp"
#include "../math/BoundingSphere.hpp"

class WrappedTorus : public WrappedSignedDistanceFunction {
private:
    BoundingSphere sphere;
    public:
    WrappedTorus(TorusDistanceFunction * function_, const Transformation &model, float bias);
    ~WrappedTorus();
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    const char* getLabel() const override;
};

 

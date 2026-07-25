#pragma once
#include "WrappedSignedDistanceFunction.hpp"
#include "BoxDistanceFunction.hpp"
#include "../math/BoundingSphere.hpp"

class WrappedBox : public WrappedSignedDistanceFunction {
private:
    BoundingSphere sphere;
    public:
    WrappedBox(BoxDistanceFunction * function_, const Transformation &model = Transformation(), float bias = 0.0f);
    ~WrappedBox();
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    const char* getLabel() const override;
};

 

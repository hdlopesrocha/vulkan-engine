#pragma once
#include "WrappedSignedDistanceFunction.hpp"
#include "PyramidDistanceFunction.hpp"
#include "../math/BoundingSphere.hpp"

class WrappedPyramid : public WrappedSignedDistanceFunction {
private:
    BoundingSphere sphere;
    public:
    WrappedPyramid(PyramidDistanceFunction * function_, const Transformation &model, float bias);
    ~WrappedPyramid();
    float boundingSphereRadius(float width, float depth, float height) const;
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    const char* getLabel() const override;
};

 

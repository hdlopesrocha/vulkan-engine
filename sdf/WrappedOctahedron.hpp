#pragma once
#include "WrappedSignedDistanceFunction.hpp"
#include "OctahedronDistanceFunction.hpp"
#include "../math/BoundingSphere.hpp"

class WrappedOctahedron : public WrappedSignedDistanceFunction {
private:
    BoundingSphere sphere;
    public:
    WrappedOctahedron(OctahedronDistanceFunction * function_, const Transformation &model = Transformation(), float bias = 0.0f);
    ~WrappedOctahedron();
    BoundingSphere getSphere(const Transformation &model, float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    const char* getLabel() const override;
};

 

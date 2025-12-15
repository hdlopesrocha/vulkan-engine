#pragma once
#include "WrappedSignedDistanceFunction.hpp"
#include "CylinderDistanceFunction.hpp"
#include "../math/BoundingSphere.hpp"

class WrappedCylinder : public WrappedSignedDistanceFunction {
    public:
    WrappedCylinder(CylinderDistanceFunction * function);
    ~WrappedCylinder();
    BoundingSphere getSphere(const Transformation &model, float bias) const;
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    bool isContained(const BoundingCube &cube, const Transformation &model, float bias) const override;
    float getLength(const Transformation &model, float bias) const override;
    void accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const override;
    const char* getLabel() const override;
};

 

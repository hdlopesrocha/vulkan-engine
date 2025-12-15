#ifndef SDF_WRAPPED_CONE_HPP
#define SDF_WRAPPED_CONE_HPP

#include "WrappedSignedDistanceFunction.hpp"
#include "ConeDistanceFunction.hpp"
#include "../math/BoundingSphere.hpp"

class WrappedCone : public WrappedSignedDistanceFunction {
public:
    WrappedCone(ConeDistanceFunction * function);
    ~WrappedCone();
    BoundingSphere getSphere(const Transformation &model, float bias) const;
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    bool isContained(const BoundingCube &cube, const Transformation &model, float bias) const override;
    float getLength(const Transformation &model, float bias) const override;
    void accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const override;
    const char* getLabel() const override;
};

#endif

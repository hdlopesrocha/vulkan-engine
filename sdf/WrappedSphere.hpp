#ifndef SDF_WRAPPED_SPHERE_HPP
#define SDF_WRAPPED_SPHERE_HPP

#include "WrappedSignedDistanceFunction.hpp"
#include "SphereDistanceFunction.hpp"
#include "../math/BoundingSphere.hpp"

class WrappedSphere : public WrappedSignedDistanceFunction {
    public:
    WrappedSphere(SphereDistanceFunction * function);
    ~WrappedSphere();
    BoundingSphere getSphere(const Transformation &model, float bias) const;
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    bool isContained(const BoundingCube &cube, const Transformation &model, float bias) const override;
    float getLength(const Transformation &model, float bias) const override;
    void accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const override;
    const char* getLabel() const override;
};

#endif

#ifndef SDF_WRAPPED_CAPSULE_HPP
#define SDF_WRAPPED_CAPSULE_HPP

#include "WrappedSignedDistanceFunction.hpp"
#include "CapsuleDistanceFunction.hpp"
#include "../math/BoundingBox.hpp"

class WrappedCapsule : public WrappedSignedDistanceFunction {
    public:
    WrappedCapsule(CapsuleDistanceFunction * function);
    ~WrappedCapsule();
    BoundingBox getBox(const Transformation &model, float bias) const;
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    bool isContained(const BoundingCube &cube, const Transformation &model, float bias) const override;
    float getLength(const Transformation &model, float bias) const override;
    void accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const override;
    const char* getLabel() const override;
};

#endif

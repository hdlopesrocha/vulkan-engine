#ifndef SDF_WRAPPED_BOX_HPP
#define SDF_WRAPPED_BOX_HPP

#include "WrappedSignedDistanceFunction.hpp"
#include "BoxDistanceFunction.hpp"
#include "../math/BoundingSphere.hpp"

class WrappedBox : public WrappedSignedDistanceFunction {
    public:
    WrappedBox(BoxDistanceFunction * function);
    ~WrappedBox();
    BoundingSphere getSphere(const Transformation &model, float bias) const;
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    bool isContained(const BoundingCube &cube, const Transformation &model, float bias) const override;
    float getLength(const Transformation &model, float bias) const override;
    void accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const override;
    const char* getLabel() const override;
};

#endif

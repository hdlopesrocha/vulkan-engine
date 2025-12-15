#ifndef SDF_WRAPPED_TORUS_HPP
#define SDF_WRAPPED_TORUS_HPP

#include "WrappedSignedDistanceFunction.hpp"
#include "TorusDistanceFunction.hpp"
#include "../math/BoundingSphere.hpp"

class WrappedTorus : public WrappedSignedDistanceFunction {
    public:
    WrappedTorus(TorusDistanceFunction * function);
    ~WrappedTorus();
    BoundingSphere getSphere(const Transformation &model, float bias) const;
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    bool isContained(const BoundingCube &cube, const Transformation &model, float bias) const override;
    float getLength(const Transformation &model, float bias) const override;
    void accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const override;
    const char* getLabel() const override;
};

#endif

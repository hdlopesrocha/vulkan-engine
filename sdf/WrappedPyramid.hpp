#ifndef SDF_WRAPPED_PYRAMID_HPP
#define SDF_WRAPPED_PYRAMID_HPP

#include "WrappedSignedDistanceFunction.hpp"
#include "PyramidDistanceFunction.hpp"
#include "../math/BoundingSphere.hpp"

class WrappedPyramid : public WrappedSignedDistanceFunction {
    public:
    WrappedPyramid(PyramidDistanceFunction * function);
    ~WrappedPyramid();
    float boundingSphereRadius(float width, float depth, float height) const;
    BoundingSphere getSphere(const Transformation &model, float bias) const;
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    bool isContained(const BoundingCube &cube, const Transformation &model, float bias) const override;
    float getLength(const Transformation &model, float bias) const override;
    void accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const override;
    const char* getLabel() const override;
};

#endif

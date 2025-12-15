#ifndef SDF_WRAPPED_HEIGHTMAP_HPP
#define SDF_WRAPPED_HEIGHTMAP_HPP

#include "WrappedSignedDistanceFunction.hpp"
#include "HeightMapDistanceFunction.hpp"
#include "../math/BoundingBox.hpp"

class WrappedHeightMap : public WrappedSignedDistanceFunction {
    public:
    WrappedHeightMap(HeightMapDistanceFunction * function);
    ~WrappedHeightMap();
    BoundingBox getBox(float bias) const;
    ContainmentType check(const BoundingCube &cube, const Transformation &model, float bias) const override;
    bool isContained(const BoundingCube &cube, const Transformation &model, float bias) const override;
    float getLength(const Transformation &model, float bias) const override;
    void accept(BoundingVolumeVisitor &visitor, const Transformation &model, float bias) const override;
    const char* getLabel() const override;
};

#endif

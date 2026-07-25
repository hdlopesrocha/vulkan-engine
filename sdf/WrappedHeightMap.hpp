#pragma once
#include "WrappedSignedDistanceFunction.hpp"
#include "HeightMapDistanceFunction.hpp"
#include "../math/BoundingBox.hpp"

class WrappedHeightMap : public WrappedSignedDistanceFunction {
private:
    BoundingBox box;
    public:
    WrappedHeightMap(HeightMapDistanceFunction * function_, float bias = 0.0f);
    ~WrappedHeightMap();
    BoundingBox getBox(float bias) const override;
    ContainmentType check(const BoundingCube &cube) const override;
    bool isContained(const BoundingCube &cube) const override;
    const char* getLabel() const override;
};

 

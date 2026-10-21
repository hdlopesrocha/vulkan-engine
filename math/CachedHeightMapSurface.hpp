#pragma once
#include "HeightFunction.hpp"
#include "geometry/BoundingBox.hpp"
#include "geometry/BoundingCube.hpp"
#include <vector>

class CachedHeightMapSurface : public HeightFunction {
public:
    std::vector<std::vector<float>> data;
    BoundingBox box;
    int width;
    int height;

    CachedHeightMapSurface(const HeightFunction &function, BoundingBox box_, float delta);
    float getData(int x, int z) const;
    float getHeightAt(float x, float z) const override;
};

 

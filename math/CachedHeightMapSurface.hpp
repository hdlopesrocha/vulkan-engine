#pragma once
#include "HeightFunction.hpp"
#include "BoundingBox.hpp"
#include "BoundingCube.hpp"
#include <vector>

class CachedHeightMapSurface : public HeightFunction {
public:
    std::vector<std::vector<float>> data;
    BoundingBox box;
    int width;
    int height;

    CachedHeightMapSurface(const HeightFunction &function, BoundingBox box, float delta);
    float getData(int x, int z) const;
    float getHeightAt(float x, float z) const override;
    glm::vec2 getHeightRangeBetween(const BoundingCube &cube, int range) const;
};

 

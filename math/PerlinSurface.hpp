#pragma once
#include "HeightFunction.hpp"

class PerlinSurface : public HeightFunction {
public:
    float amplitude;
    float frequency;
    float offset;

    PerlinSurface(float amplitude_, float frequency_, float offset_);
    float getHeightAt(float x, float z) const override;
};

 

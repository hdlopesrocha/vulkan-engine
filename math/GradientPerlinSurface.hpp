#pragma once
#include "PerlinSurface.hpp"

class GradientPerlinSurface : public PerlinSurface {
public:
    GradientPerlinSurface(float amplitude_, float frequency_, float offset_);
    float getHeightAt(float x, float z) const override;
};

 

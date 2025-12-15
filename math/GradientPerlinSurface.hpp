#pragma once
#include "PerlinSurface.hpp"

class GradientPerlinSurface : public PerlinSurface {
public:
    GradientPerlinSurface(float amplitude, float frequency, float offset);
    float getHeightAt(float x, float z) const override;
};

 

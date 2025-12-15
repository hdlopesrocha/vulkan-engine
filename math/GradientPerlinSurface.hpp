#ifndef GRADIENT_PERLIN_SURFACE_HPP
#define GRADIENT_PERLIN_SURFACE_HPP

#include "PerlinSurface.hpp"

class GradientPerlinSurface : public PerlinSurface {
public:
    GradientPerlinSurface(float amplitude, float frequency, float offset);
    float getHeightAt(float x, float z) const override;
};

#endif

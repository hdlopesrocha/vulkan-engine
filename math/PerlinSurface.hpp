#ifndef PERLIN_SURFACE_HPP
#define PERLIN_SURFACE_HPP

#include "HeightFunction.hpp"

class PerlinSurface : public HeightFunction {
public:
    float amplitude;
    float frequency;
    float offset;

    PerlinSurface(float amplitude, float frequency, float offset);
    float getHeightAt(float x, float z) const override;
};

#endif

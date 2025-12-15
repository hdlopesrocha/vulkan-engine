#ifndef FRACTAL_PERLIN_SURFACE_HPP
#define FRACTAL_PERLIN_SURFACE_HPP

#include "PerlinSurface.hpp"

class FractalPerlinSurface : public PerlinSurface {
public:
    using PerlinSurface::PerlinSurface;
    FractalPerlinSurface(float amplitude, float frequency, float offset);
    float getHeightAt(float x, float z) const override;
};

#endif


#include "PerlinSurface.hpp"
#include "Math.hpp"
#include <stb/stb_perlin.h>



PerlinSurface::PerlinSurface(float amplitude_, float frequency_, float offset_) {
    this->amplitude = amplitude_;
    this->frequency = frequency_;
    this->offset = offset_;
}

float PerlinSurface::getHeightAt(float x, float z) const {

    float noise = stb_perlin_noise3(x* frequency, 0, z*frequency, 0, 0, 0);
    noise = offset + amplitude * noise;
    return noise;
}

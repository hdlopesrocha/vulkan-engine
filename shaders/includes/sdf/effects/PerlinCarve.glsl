#ifndef SDF_EFFECT_PERLIN_CARVE_GLSL
#define SDF_EFFECT_PERLIN_CARVE_GLSL

// CPU PerlinCarveDistanceEffect mirror: positive-only thresholded fractal
// noise that is added to the distance field.
float sdfEffectCarveFractal(vec3 p, float threshold, float frequency) {
    float noiseValue = 0.0;
    float freq = frequency;
    float amp = 1.0;
    for (int i = 0; i < 6; ++i) {
        noiseValue += amp * perlinNoise3D(p * freq);
        freq *= 2.0;
        amp *= 0.5;
    }
    if (noiseValue > threshold) return noiseValue - threshold;
    return 0.0;
}

#endif // SDF_EFFECT_PERLIN_CARVE_GLSL

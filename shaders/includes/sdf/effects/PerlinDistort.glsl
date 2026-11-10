#ifndef SDF_EFFECT_PERLIN_DISTORT_GLSL
#define SDF_EFFECT_PERLIN_DISTORT_GLSL

// CPU PerlinDistortDistanceEffect mirror: six-octave vector Perlin field used
// as a domain warp (the brush then evaluates the mixed field at localP + amp*n).
vec3 sdfEffectPerlinDistort(vec3 p, float frequency, float brightness, float contrast) {
    vec3 total = vec3(0.0);
    float freq = frequency;
    float amp = 1.0;
    for (int i = 0; i < 6; ++i) {
        total += amp * vec3(
            perlinNoise3D(p * freq + vec3(0.0)),
            perlinNoise3D(p * freq + vec3(100.0)),
            perlinNoise3D(p * freq + vec3(200.0)));
        freq *= 2.0;
        amp *= 0.5;
    }
    return vec3(sdfEffectBc(total.x, brightness, contrast),
                sdfEffectBc(total.y, brightness, contrast),
                sdfEffectBc(total.z, brightness, contrast));
}

#endif // SDF_EFFECT_PERLIN_DISTORT_GLSL

#ifndef SDF_EFFECT_SINE_DISTORT_GLSL
#define SDF_EFFECT_SINE_DISTORT_GLSL

// CPU SineDistortDistanceEffect mirror: three separable sine/cosine products
// used as a domain warp (the brush evaluates the mixed field at the warped
// point and divides by the Lipschitz scale).
vec3 sdfEffectSineDisplacement(vec3 p, float frequency) {
    float dx = sin(p.x * frequency) * cos(p.y * frequency) * sin(p.z * frequency);
    float dy = cos(p.x * frequency) * sin(p.y * frequency) * cos(p.z * frequency);
    float dz = sin(p.x * frequency) * sin(p.y * frequency) * cos(p.z * frequency);
    return vec3(dx, dy, dz);
}

#endif // SDF_EFFECT_SINE_DISTORT_GLSL

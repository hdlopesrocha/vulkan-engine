#ifndef SMOKE_G_P_U_GLSL
#define SMOKE_G_P_U_GLSL

// Extracted from shaders/includes/sdf_smoke.glsl (single-struct GLSL type).

// Mirrors vulkan/types/SmokeGPU.hpp (std430): SmokeGPU = 11 vec4
// (8 smoke + 3 tracer gold), BulletGPU = 3 vec4.
struct SmokeGPU {
    vec4 timing;    // x = growth duration (s), y = loop duration (s),
                    // z = dissipation, w = unused
    vec4 noise;     // x = noise scale (1/m), y = noise strength,
                    // z = warp strength, w = unused
    vec4 wind;      // xy = wind velocity (m/s), z = live density scale, w = unused
    vec4 tunnel;    // x = tunnel strength, y = tunnel falloff (m),
                    // z = wake strength, w = wake dissipation = refill rate (1/s)
    vec4 wake;      // x = wake radius (m), y = wake expansion,
                    // z = wake length (m), w = unused
    vec4 pressure;  // x = pressure radius (m), y = pressure strength,
                    // z = wave speed (m/s), w = wave frequency (1/m)
    vec4 turbWave;  // x = wave falloff (1/m), y = turbulence scale (1/m),
                    // z = turbulence strength, w = turbulence speed (1/s)
    vec4 render;    // x = shadow samples, y = shadow strength, zw unused
    vec4 gold0;     // rgb = deep gold, w = specular power
    vec4 gold1;     // rgb = bright/champagne gold, w = pattern scale
    vec4 gold2;     // x = specular strength, y = fresnel boost,
                    // z = warm floor, w = normal distortion
};

#endif // SMOKE_G_P_U_GLSL

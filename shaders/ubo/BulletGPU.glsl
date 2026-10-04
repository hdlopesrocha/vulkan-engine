#ifndef BULLET_G_P_U_GLSL
#define BULLET_G_P_U_GLSL

// Extracted from shaders/includes/sdf_smoke.glsl (single-struct GLSL type).

struct BulletGPU {
    vec4 a; // xyz = path start (world), w = radiusStart (m)
    vec4 b; // xyz = velocity (m/s; magnitude = speed), w = path length (m)
    vec4 c; // x = radiusEnd (m), y = loopDuration (s), z = intensity (0 = empty), w = phase (s)
};

#endif // BULLET_G_P_U_GLSL

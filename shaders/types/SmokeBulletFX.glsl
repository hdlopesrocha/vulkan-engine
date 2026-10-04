#ifndef SMOKE_BULLET_F_X_GLSL
#define SMOKE_BULLET_F_X_GLSL

// Extracted from shaders/includes/sdf_smoke.glsl (single-struct GLSL type).

// Bullet interaction field at p (geometry only, no noise — cheap): tunnel
// thinning [0,1], domain displacement, shock-wave value, wake and
// turbulence magnitudes (for debug views). Anisotropic by construction
// (§28): compression ahead of the head, radial push at the sides,
// expanding turbulent wake behind.
struct SmokeBulletFX {
    float thin;      // multiplicative density removal (tunnel core)
    vec3 displace;   // noise-domain displacement (pressure + wake + swirl)
    float wave;      // shock-wave modulation value
    float wake;      // wake influence (debug + thinning)
    float turb;      // turbulence magnitude (debug)
};

#endif // SMOKE_BULLET_F_X_GLSL

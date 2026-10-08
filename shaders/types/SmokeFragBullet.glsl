#ifndef SMOKE_FRAG_BULLET_GLSL
#define SMOKE_FRAG_BULLET_GLSL

// Canonical smoke state block (std430, 624 B). Single definition shared with
// the CPU: sdf/types/SmokeFragBullet.hpp — identical layout and offsets.
// Smoke is 16-byte aligned, Bullet has a 48-byte stride and the instance
// transform follows the bullets with std430 vec3 16-byte alignment:
//   tuning 0, bullets 176 (stride 48), worldPos 560, worldScale 572,
//   rotCol0 576, rotCol1 592, rotCol2 608, tracerActive 620.
#include "Bullet.glsl"
#include "Smoke.glsl"

struct SmokeFragBullet {
    Smoke tuning;     // offset 0
    Bullet bullets[8];// offset 176, stride 48
    // World transform of the instance whose LOCAL frame the bullets (and
    // wind) are packed in: the smoke bomb's generic SdfModel. rotCol0..2 are
    // the world-rotation matrix columns, so local = (dot(rotCol0, v),
    // dot(rotCol1, v), dot(rotCol2, v)) for a world vector v (no
    // translation). The analytic gold-tracer test runs in this frame.
    vec3 worldPos;    // offset 560  instance translation
    float worldScale; // offset 572  uniform scale (1 for the smoke bomb)
    vec3 rotCol0;     // offset 576
    vec3 rotCol1;     // offset 592
    vec3 rotCol2;     // offset 608
    float tracerActive;// offset 620  0 = smoke disabled / Fire shape
};

#endif // SMOKE_FRAG_BULLET_GLSL

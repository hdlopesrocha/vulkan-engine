#ifndef SMOKE_FRAG_BULLET_GLSL
#define SMOKE_FRAG_BULLET_GLSL

// Canonical smoke state block (std430, 560 B). Single definition shared with
// the CPU: sdf/types/SmokeFragBullet.hpp — identical layout and offsets.
// Smoke is 16-byte aligned and Bullet has a 48-byte stride, so the
// nested layout is exactly tuning then bullets[8].
#include "Bullet.glsl"
#include "Smoke.glsl"

struct SmokeFragBullet {
    Smoke tuning;     // offset 0
    Bullet bullets[8];// offset 176, stride 48
};

#endif // SMOKE_FRAG_BULLET_GLSL

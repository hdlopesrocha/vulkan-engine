#pragma once

// Canonical smoke state block — ONE definition shared by the CPU (uploaded
// verbatim into the set=1 binding 8 SSBO) and the GPU shaders (GLSL twin:
// shaders/types/SmokeFragBullet.glsl, identical layout). std430;
// Smoke is 16-byte aligned and Bullet has a 48-byte stride, so the
// nested layout is exactly tuning then bullets[8], size 560.
#include <cstddef>

#include "sdf/types/Bullet.hpp"
#include "sdf/types/Smoke.hpp"

struct SmokeFragBullet {
    Smoke tuning;                          // offset 0
    Bullet bullets[kSmokeMaxBullets];      // offset 176, stride 48
};
static_assert(sizeof(SmokeFragBullet) == 176 + 8 * 48, "SmokeFragBullet size mismatch");
static_assert(offsetof(SmokeFragBullet, tuning) == 0, "tuning offset");
static_assert(offsetof(SmokeFragBullet, bullets) == 176, "bullets offset");

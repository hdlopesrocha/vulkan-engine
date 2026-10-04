#pragma once

// Smoke-bomb runtime state (set=1 binding 8, SmokeBlock): tuning + bullets.
// Updated on demand (widget tweaks, bullet fire), never per frame: time
// comes from the shared params UBO, bullet motion/refill are GPU functions
// of it.
#include "SmokeGPU.hpp"
#include "BulletGPU.hpp"

struct SmokeFragBulletGPU {
    SmokeGPU tuning;
    BulletGPU bullets[kSmokeMaxBullets];
};
static_assert(sizeof(SmokeFragBulletGPU) == 128 + 8 * 48, "SmokeFragBulletGPU size mismatch");

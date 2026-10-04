#pragma once

// One bullet interaction (std430, 48 bytes). The bullet is a looping
// projectile: it restarts at `start` with the given velocity every
// `loopDuration` seconds, offset in time by `phase` (so several bullets
// can be staggered, and the auto bullet can wait for the smoke bloom).
// All motion, aging and refill are GPU functions of global time; in-flight
// bullets need no CPU updates.
//
// The carved smoke volume is a capped capsule (round cone) swept along the
// traveled path, with the radius tapering from `radiusStart` (launch) to
// `radiusEnd` (at the head).
//
//   a = (path start xyz, radiusStart)
//   b = (velocity xyz in m/s; magnitude = speed, path length)
//   c = (radiusEnd, loopDuration [s], intensity (0 = empty), phase [s])
#include <glm/glm.hpp>
#include <cstdint>

struct BulletGPU {
    glm::vec4 a;
    glm::vec4 b;
    glm::vec4 c;
};
static_assert(sizeof(BulletGPU) == 48, "BulletGPU must be 48 bytes");
static_assert(sizeof(BulletGPU) % 16 == 0, "BulletGPU must be multiple of 16");
inline constexpr uint32_t kSmokeMaxBullets = 8;

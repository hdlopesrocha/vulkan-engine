#pragma once

// Canonical bullet interaction state — ONE definition shared by the CPU
// (fire/repack) and the GPU shaders (GLSL twin: shaders/types/Bullet.glsl,
// identical layout). Lives in the SmokeFragBullet state block (set=1
// binding 8, SmokeBlock, std430; array stride = sizeof == 48).
//
// start/velocity are conceptual vectors; the radii, path length, loop
// timing and intensity are independent scalars. start/velocity are stored in
// the smoke shape's LOCAL frame (the generic SdfModel owns the transform).
//
// Member alignment is pinned with alignas to the std430 vector rules
// (vec3 -> 16) so the layout never depends on glm packing.
#include <cstddef>
#include <cstdint>

#include <glm/glm.hpp>

struct Bullet {
    alignas(16) glm::vec3 start{0.0f};   // offset  0  path start (smoke-local)
    float radiusStart = 0.0f;            // offset 12  radius at launch (m)
    alignas(16) glm::vec3 velocity{0.0f};// offset 16  velocity (m/s, smoke-local; magnitude = speed)
    float pathLength = 0.0f;             // offset 28  carved path length (m)
    float radiusEnd = 0.0f;              // offset 32  radius at the head (m)
    float loopDuration = 0.0f;           // offset 36  per-bullet cycle (s; 0 = one-shot)
    float intensity = 0.0f;              // offset 40  0 = empty slot
    float phase = 0.0f;                  // offset 44  launch time within the loop (s)
};
static_assert(sizeof(Bullet) == 48, "Bullet must be 48 bytes");
static_assert(offsetof(Bullet, start) == 0, "start offset");
static_assert(offsetof(Bullet, radiusStart) == 12, "radiusStart offset");
static_assert(offsetof(Bullet, velocity) == 16, "velocity offset");
static_assert(offsetof(Bullet, pathLength) == 28, "pathLength offset");
static_assert(offsetof(Bullet, radiusEnd) == 32, "radiusEnd offset");
static_assert(offsetof(Bullet, loopDuration) == 36, "loopDuration offset");
static_assert(offsetof(Bullet, intensity) == 40, "intensity offset");
static_assert(offsetof(Bullet, phase) == 44, "phase offset");

inline constexpr uint32_t kSmokeMaxBullets = 8u;

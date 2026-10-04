#pragma once

// One bullet interaction (std430, 48 bytes). Slot 0 is the auto-loop template
// (reborn every loop from widget params); slots 1..7 are manual (birth set
// by Fire, intensity 0 = empty). Motion, wake age and refill are GPU-side
// functions of global time, so in-flight bullets need no CPU updates.
#include <glm/glm.hpp>
#include <cstdint>

struct BulletGPU {
    glm::vec4 a; // xyz = path start (world), w = tunnel radius
    glm::vec4 b; // xyz = direction (unit), w = path length
    glm::vec4 c; // x = speed (m/s), y = birth time (s), z = intensity, w = flags
};
static_assert(sizeof(BulletGPU) == 48, "BulletGPU must be 48 bytes");
static_assert(sizeof(BulletGPU) % 16 == 0, "BulletGPU must be multiple of 16");
inline constexpr uint32_t kSmokeMaxBullets = 8;

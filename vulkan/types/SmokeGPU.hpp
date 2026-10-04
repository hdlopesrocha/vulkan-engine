#pragma once

// Smoke-bomb runtime tuning (std430, 128 bytes). Lives in the smoke state
// buffer (set=1 binding 8); streamed on demand, never rebuilt with geometry.
#include <glm/glm.hpp>
#include <cstdint>

struct SmokeGPU {
    glm::vec4 timing;   // x = growth duration (s), y = loop duration (s),
                        // z = dissipation, w = unused
    glm::vec4 noise;    // x = noise scale, y = noise strength,
                        // z = warp strength, w = unused
    glm::vec4 wind;     // xy = wind velocity (m/s), z = live density scale, w = unused
    glm::vec4 tunnel;   // x = tunnel strength, y = tunnel falloff,
                        // z = wake strength, w = wake dissipation (refill rate)
    glm::vec4 wake;     // x = wake radius, y = wake expansion, z = wake length, w = unused
    glm::vec4 pressure; // x = pressure radius, y = pressure strength,
                        // z = wave speed, w = wave frequency
    glm::vec4 turbWave; // x = wave falloff, y = turbulence scale,
                        // z = turbulence strength, w = turbulence speed
    glm::vec4 render;   // x = shadow samples, y = shadow strength, zw unused
};
static_assert(sizeof(SmokeGPU) == 128, "SmokeGPU must be 128 bytes");

#pragma once

// Smoke-bomb runtime tuning (std430, 176 bytes). Lives in the smoke state
// buffer (set=1 binding 8); streamed on demand, never rebuilt with geometry.
// Layout: 8 core vec4 (smoke + bullet simulation) followed by 3 gold vec4
// for the bullet tracer shading (moved here from shader constants so the
// widget can edit the gold look).
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
    glm::vec4 gold0;    // rgb = deep gold, w = specular power
    glm::vec4 gold1;    // rgb = bright/champagne gold, w = pattern scale
    glm::vec4 gold2;    // x = specular strength, y = fresnel boost,
                        // z = warm floor, w = normal distortion
};
static_assert(sizeof(SmokeGPU) == 176, "SmokeGPU must be 176 bytes");

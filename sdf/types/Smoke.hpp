#pragma once

// Canonical smoke-bomb tuning — ONE definition shared by the CPU (widget /
// renderer setters) and the GPU shaders (GLSL twin: shaders/types/Smoke.glsl,
// identical layout). Lives in the SmokeFragBullet state block (set=1
// binding 8, SmokeBlock, std430; array stride = sizeof == 176).
//
// wind is one conceptual vector (local-frame velocity); smokeColor/goldDeep/
// goldBright are conceptual colors, always followed by one independent
// scalar partner (density scale / specular power / pattern scale) because
// the std430 vector rules legally pack them. Every other value is an
// independent scalar property.
//
// Member alignment is pinned with alignas to the std430 vector rules
// (vec2 -> 8, vec3 -> 16) so the layout never depends on glm packing.
#include <cstddef>
#include <cstdint>

#include <glm/glm.hpp>

struct Smoke {
    float growthDuration = 1.0f;   // offset   0  expansion window (s)
    float loopDuration = 10.0f;    // offset   4  animation repeat period (s)
    float dissipation = 0.15f;     // offset   8  end-of-loop fade strength
    float noiseScale = 0.05f;      // offset  12  1/m
    float noiseStrength = 1.0f;    // offset  16
    float noiseWarp = 0.6f;        // offset  20
    alignas(8) glm::vec2 wind{0.0f}; // offset 24  local-frame wind velocity (m/s)
    float densityScale = 1.0f;     // offset  32  live density multiplier
    float tunnelStrength = 0.9f;   // offset  36
    float tunnelFalloff = 12.0f;   // offset  40  m
    float wakeStrength = 0.8f;     // offset  44
    float wakeDissipation = 1.5f;  // offset  48  1/s, doubles as tunnel refill rate
    float wakeRadius = 12.0f;      // offset  52  m
    float wakeExpansion = 0.15f;   // offset  56  wake entrainment coefficient
    float wakeLength = 120.0f;     // offset  60  m
    float pressureRadius = 64.0f;  // offset  64  m
    float pressureStrength = 6.0f; // offset  68  /4 = reference shock push
    float rippleAmp = 0.05f;       // offset  72  compression-turbulence ripple amplitude
    float rippleFreq = 9.0f;       // offset  76  ripple bands per local bore radius
    float turbScale = 0.08f;       // offset  80  1/m
    float turbStrength = 2.5f;     // offset  84
    float turbSpeed = 1.5f;        // offset  88  1/s
    uint32_t shadowSamples = 3u;   // offset  92
    float shadowStrength = 0.8f;   // offset  96
    float heatStrength = 0.0f;     // offset 100  hot-air emission multiplier
    uint32_t shape = 0u;           // offset 104  0 cloud, 1 sphere, 2 cube (Fire builds a flame instance)
    // offset 108..112 is padding: smokeColor must start at a 16-byte boundary.
    alignas(16) glm::vec3 smokeColor{0.557f, 0.635f, 0.784f}; // offset 112  lit albedo tint
    // offset 124..128 is padding: goldDeep must start at a 16-byte boundary.
    alignas(16) glm::vec3 goldDeep{0.45f, 0.22f, 0.05f};      // offset 128  tracer deep gold
    float goldSpecPower = 64.0f;   // offset 140
    alignas(16) glm::vec3 goldBright{1.0f, 0.85f, 0.55f};     // offset 144  tracer bright gold
    float goldPatternScale = 0.8f; // offset 156
    float goldSpecStrength = 2.0f; // offset 160
    float goldFresnelBoost = 0.6f; // offset 164
    float goldWarmFloor = 0.35f;   // offset 168
    float goldNormalDistort = 0.6f;// offset 172
    // offset 176 is the struct end (multiple of 16; no tail padding needed).
};
static_assert(sizeof(Smoke) == 176, "Smoke must be 176 bytes");
static_assert(offsetof(Smoke, growthDuration) == 0, "growthDuration offset");
static_assert(offsetof(Smoke, loopDuration) == 4, "loopDuration offset");
static_assert(offsetof(Smoke, dissipation) == 8, "dissipation offset");
static_assert(offsetof(Smoke, noiseScale) == 12, "noiseScale offset");
static_assert(offsetof(Smoke, noiseStrength) == 16, "noiseStrength offset");
static_assert(offsetof(Smoke, noiseWarp) == 20, "noiseWarp offset");
static_assert(offsetof(Smoke, wind) == 24, "wind offset");
static_assert(offsetof(Smoke, densityScale) == 32, "densityScale offset");
static_assert(offsetof(Smoke, tunnelStrength) == 36, "tunnelStrength offset");
static_assert(offsetof(Smoke, tunnelFalloff) == 40, "tunnelFalloff offset");
static_assert(offsetof(Smoke, wakeStrength) == 44, "wakeStrength offset");
static_assert(offsetof(Smoke, wakeDissipation) == 48, "wakeDissipation offset");
static_assert(offsetof(Smoke, wakeRadius) == 52, "wakeRadius offset");
static_assert(offsetof(Smoke, wakeExpansion) == 56, "wakeExpansion offset");
static_assert(offsetof(Smoke, wakeLength) == 60, "wakeLength offset");
static_assert(offsetof(Smoke, pressureRadius) == 64, "pressureRadius offset");
static_assert(offsetof(Smoke, pressureStrength) == 68, "pressureStrength offset");
static_assert(offsetof(Smoke, rippleAmp) == 72, "rippleAmp offset");
static_assert(offsetof(Smoke, rippleFreq) == 76, "rippleFreq offset");
static_assert(offsetof(Smoke, turbScale) == 80, "turbScale offset");
static_assert(offsetof(Smoke, turbStrength) == 84, "turbStrength offset");
static_assert(offsetof(Smoke, turbSpeed) == 88, "turbSpeed offset");
static_assert(offsetof(Smoke, shadowSamples) == 92, "shadowSamples offset");
static_assert(offsetof(Smoke, shadowStrength) == 96, "shadowStrength offset");
static_assert(offsetof(Smoke, heatStrength) == 100, "heatStrength offset");
static_assert(offsetof(Smoke, shape) == 104, "shape offset");
static_assert(offsetof(Smoke, smokeColor) == 112, "smokeColor offset");
static_assert(offsetof(Smoke, goldDeep) == 128, "goldDeep offset");
static_assert(offsetof(Smoke, goldSpecPower) == 140, "goldSpecPower offset");
static_assert(offsetof(Smoke, goldBright) == 144, "goldBright offset");
static_assert(offsetof(Smoke, goldPatternScale) == 156, "goldPatternScale offset");
static_assert(offsetof(Smoke, goldSpecStrength) == 160, "goldSpecStrength offset");
static_assert(offsetof(Smoke, goldFresnelBoost) == 164, "goldFresnelBoost offset");
static_assert(offsetof(Smoke, goldWarmFloor) == 168, "goldWarmFloor offset");
static_assert(offsetof(Smoke, goldNormalDistort) == 172, "goldNormalDistort offset");

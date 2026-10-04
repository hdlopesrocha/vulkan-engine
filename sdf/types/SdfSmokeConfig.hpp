#pragma once

// Smoke-bomb tuning (see SmokeBulletWidget sections).
#include <glm/glm.hpp>
#include <cstdint>

struct SdfSmokeConfig {
    bool enabled = true;
    glm::vec3 pos = glm::vec3(0.0f, 1400.0f, 0.0f);
    float radius = 256.0f;     // maximum radius, marching bounds (m)
    float seed = 0.0f;
    float density = 2.0f;      // base material density (rebuilds scene)
    float absorption = 0.75f;   // (rebuilds scene)
    float scattering = 8.0f;   // (rebuilds scene)
    float growthDuration = 2.0f;  // rapid expansion time (s)
    float loopDuration = 10.0f;   // animation repeat period (s)
    float dissipation = 0.15f;
    float noiseScale = 256.0f;    // 1/m
    float noiseStrength = 1.0f;
    float noiseWarp = 0.6f;
    float windSpeed = 16.0f;       // m/s
    float windAngleDeg = 45.0f;   // XZ plane, 0 = +X
    float densityScale = 1.0f;    // live multiplier (no rebuild)
    float tunnelStrength = 0.9f;
    float tunnelFalloff = 12.0f;  // m
    float wakeStrength = 0.8f;
    float wakeRadius = 10.0f;     // m
    float wakeExpansion = 0.15f;
    float wakeLength = 120.0f;    // m
    float wakeDissipation = 1.5f; // 1/s, doubles as tunnel refill rate
    float pressureRadius = 18.0f; // m
    float pressureStrength = 6.0f;
    float pressureWaveSpeed = 60.0f;  // m/s
    float pressureWaveFreq = 0.35f;   // 1/m
    float pressureWaveFalloff = 0.05f;// 1/m
    float turbScale = 0.08f;         // 1/m
    float turbStrength = 2.5f;
    float turbSpeed = 1.5f;          // 1/s
    int shadowSamples = 3;
    float shadowStrength = 0.8f;
    uint32_t debugView = 0; // 0 = normal smoke, 1-10 per spec §24
};

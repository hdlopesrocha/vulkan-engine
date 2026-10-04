#pragma once

// Bullet defaults stamped into SSBO slots on fire (slot 0 = auto-loop
// template rebuilt live from these every loop).
//
// ABI matches vulkan/types/BulletGPU.hpp: one BulletGPU slot carries
//   a = (path start xyz, radiusStart)
//   b = (velocity xyz m/s, path length)
//   c = (radiusEnd, loopDuration s, intensity 0=empty, phase s)
// so the carved volume is a tapered capsule from radiusStart (launch) to
// radiusEnd (at the head), restarted every loopDuration seconds.
//
// Also owns all bullet-movement/FX tuning streamed through the smoke SSBO
// (widget sections Bullet/Pressure/Wake/Turbulence) and the gold tracer
// shading (widget "Gold" section, smoke SSBO gold0/1/2).
#include <glm/glm.hpp>

struct SdfBulletConfig {
    // Tapered bore (capped capsule): 8 m at launch/entry, widening to 128 m
    // at the head/exit. The head bore is deliberately huge — it erases the
    // cloud around the head each pass and refill heals it behind.
    float radius = 8.0f;       // initial (launch) radius (m)
    float finalRadius = 128.0f;  // radius at the head (m)
    float loopDuration = 10.0f; // per-bullet cycle (s)
    float speed = 256.0f;         // flight speed (m/s)
    float length = 2048.0f;      // path length (m)
    float angleDeg = 0.0f;      // XZ plane, 0 = +X
    bool autoFire = true;       // re-fire from defaults every smoke loop

    // ── Bullet carve / wake / pressure / turbulence ──
    float tunnelStrength = 0.9f;
    float tunnelFalloff = 12.0f;  // m
    float wakeStrength = 0.8f;
    float wakeRadius = 12.0f;     // m (widget range 0.5-100; 256 m swallowed the ball)
    float wakeExpansion = 0.15f;
    float wakeLength = 120.0f;    // m (widget range 1-500; 1024 m churned the whole sky)
    float wakeDissipation = 1.5f; // 1/s, doubles as tunnel refill rate
    float pressureRadius = 96.0f; // m (covers the 128 m head bore wall so the rim piles outward)
    float pressureStrength = 6.0f;
    float pressureWaveSpeed = 60.0f;  // m/s
    float pressureWaveFreq = 0.35f;   // 1/m
    float pressureWaveFalloff = 0.05f;// 1/m
    float turbScale = 0.08f;         // 1/m
    float turbStrength = 2.5f;
    float turbSpeed = 1.5f;          // 1/s

    // ── Gold tracer shading (Defaults match the old shader constants) ──
    glm::vec3 goldDeep{0.45f, 0.22f, 0.05f};
    glm::vec3 goldBright{1.0f, 0.85f, 0.55f};
    float goldSpecPower = 64.0f;
    float goldSpecStrength = 2.0f;
    float goldFresnelBoost = 0.6f;
    float goldWarmFloor = 0.35f;
    float goldPatternScale = 0.8f;
    float goldNormalDistort = 0.6f;
};

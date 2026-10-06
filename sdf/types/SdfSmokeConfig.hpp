#pragma once

// Smoke-cloud tuning (see RaymarchWidget "Smoke" section). Bullet
// movement/FX (tunnel, wake, pressure, turbulence) and gold tracer shading
// live in SdfBulletConfig; both stream through the smoke SSBO.
#include <glm/glm.hpp>
#include <cstdint>

struct SdfSmokeConfig {
    bool enabled = true;
    glm::vec3 pos = glm::vec3(0.0f, 2048.0f, 0.0f); // translation
    float scale = 128.0f;      // master scale: growth size / shape size (m)
    // Volume shape (reference rig): Cloud keeps the billowy sphere domain,
    // Sphere gets a crisp dense-ball envelope, Cube a rounded cube whose
    // bounding sphere equals `scale` (half extent = scale/sqrt(3)), so the
    // CPU AABB (sphere) and the container stay valid under any rotation.
    int shape = 0;             // 0 = cloud, 1 = sphere, 2 = cube
    float yawDeg = 0.0f;       // rotation (Ry then Rx then Rz)
    float pitchDeg = 0.0f;
    float rollDeg = 0.0f;
    float seed = 0.0f;
    float density = 1.0f;      // base material density (rebuilds scene)
    float absorption = 0.9f;   // (rebuilds scene)
    float scattering = 4.0f;   // (rebuilds scene)
    float growthDuration = 1.0f;  // expansion window (s): quick start, decelerating
    float loopDuration = 10.0f;   // animation repeat period (s)
    float dissipation = 0.15f;
    // 0.05 1/m = ~20 m features on the 256 m cloud. At the old 256 1/m the
    // features were ~4 mm, which alias against any sane march step and
    // rendered as a grey wall with streak artifacts; part of the artifact fix.
    // (Must stay inside the widget's 0.001-0.2 slider range; larger values
    // alias the medium/fine octave bands into broken chunks.)
    float noiseScale = 0.05f;     // 1/m
    float noiseStrength = 1.0f;
    float noiseWarp = 0.6f;
    float windSpeed = 16.0f;       // m/s
    float windAngleDeg = 45.0f;   // XZ plane, 0 = +X
    float densityScale = 1.0f;    // live multiplier (no rebuild)
    int shadowSamples = 3;
    float shadowStrength = 0.8f;
    glm::vec3 smokeColor{0.557f, 0.635f, 0.784f}; // lit albedo tint (steel blue)
    uint32_t debugView = 0; // 0 = normal smoke, 1-10 per spec §24
};

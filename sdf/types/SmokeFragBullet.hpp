#pragma once

// Canonical smoke state block — ONE definition shared by the CPU (uploaded
// verbatim into the set=1 binding 8 SSBO) and the GPU shaders (GLSL twin:
// shaders/types/SmokeFragBullet.glsl, identical layout). std430;
// Smoke is 16-byte aligned and Bullet has a 48-byte stride, so the
// nested layout is exactly tuning then bullets[8], then the instance
// transform the bullets are packed in, then the H4 tracer sphere, size 656.
#include <cstddef>

#include "sdf/types/Bullet.hpp"
#include "sdf/types/Smoke.hpp"

struct SmokeFragBullet {
    Smoke tuning;                          // offset 0
    Bullet bullets[kSmokeMaxBullets];      // offset 176, stride 48
    // World transform of the SDF instance whose LOCAL frame the bullets
    // above (and the wind) are packed in: the smoke bomb's generic SdfModel.
    // worldPos = instance translation, worldScale = uniform scale (1 for the
    // smoke bomb; its `scale` is baked into params0/container instead).
    // rotCol0..2 are the world-rotation matrix columns, so a world vector v
    // maps to local as (dot(rotCol0, v), dot(rotCol1, v), dot(rotCol2, v)).
    // The shader runs its analytic gold-tracer ray test in this frame;
    // without it the test would mix world camera and local bullet spaces.
    // tracerActive is 0 while the smoke is disabled or on the Fire shape.
    alignas(16) glm::vec3 worldPos{0.0f};   // offset 560
    float worldScale = 1.0f;                // offset 572
    alignas(16) glm::vec3 rotCol0{1.0f, 0.0f, 0.0f}; // offset 576
    alignas(16) glm::vec3 rotCol1{0.0f, 1.0f, 0.0f}; // offset 592
    alignas(16) glm::vec3 rotCol2{0.0f, 0.0f, 1.0f}; // offset 608
    float tracerActive = 0.0f;              // offset 620
    // H4: CPU-compacted tracer sphere (SdfRenderer::refreshTracerLocked, per
    // frame): xyz = head center in the smoke-local frame above, w = radius
    // (m). meta.x = 1 while a live head-on-path round exists, 0 = no tracer
    // this frame (the fragment then pays one cached load + uniform branch and
    // zero bullet ALU). Single-flight (auto XOR manual) guarantees at most one
    // live round, so one sphere is exact.
    alignas(16) glm::vec4 tracerSphere{0.0f}; // offset 624  (xyz center, w radius)
    alignas(16) glm::vec4 tracerMeta{0.0f};   // offset 640  (x = valid)
};
static_assert(sizeof(SmokeFragBullet) == 656, "SmokeFragBullet size mismatch");
static_assert(offsetof(SmokeFragBullet, tuning) == 0, "tuning offset");
static_assert(offsetof(SmokeFragBullet, bullets) == 176, "bullets offset");
static_assert(offsetof(SmokeFragBullet, worldPos) == 560, "worldPos offset");
static_assert(offsetof(SmokeFragBullet, worldScale) == 572, "worldScale offset");
static_assert(offsetof(SmokeFragBullet, rotCol0) == 576, "rotCol0 offset");
static_assert(offsetof(SmokeFragBullet, rotCol1) == 592, "rotCol1 offset");
static_assert(offsetof(SmokeFragBullet, rotCol2) == 608, "rotCol2 offset");
static_assert(offsetof(SmokeFragBullet, tracerActive) == 620, "tracerActive offset");
static_assert(offsetof(SmokeFragBullet, tracerSphere) == 624, "tracerSphere offset");
static_assert(offsetof(SmokeFragBullet, tracerMeta) == 640, "tracerMeta offset");

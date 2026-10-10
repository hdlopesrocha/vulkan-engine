#pragma once

// Global raymarch / debug parameters (one small UBO, std140, 80 bytes).
// Canonical definition shared with the GLSL twin
// shaders/ubo/SdfParamsUBO.glsl — same names, fields and offsets.
// Independent scalars; renderMode/debugFlags are separate integers (no
// packed float). alignas(16) matches the std140 struct base alignment, so the
// block size is 80 (68 bytes of fields + 16-byte alignment rounding).
// M12 (perf report 25): smokeSamples tiers the phase-B smoke resolve
// (Settings::sdfSmokeSamples, 4..12, default 12 = reference); maxSteps and
// raycastPixelSize tier the march the same way. All three stream through the
// existing setters with no idle and no rebuild.
#include <cstddef>
#include <cstdint>

struct alignas(16) SdfParamsUBO {
    float time = 0.0f;        // offset  0  global SDF time (s)
    float maxSteps = 64.0f;   // offset  4  march step budget
    float safety = 0.7f;      // offset  8  sphere-trace step safety
    float minStep = 0.05f;    // offset 12  minimum march step
    float maxStep = 32.0f;    // offset 16  maximum march step
    float epsilon = 0.01f;    // offset 20  hit epsilon
    float earlyTerm = 0.99f;  // offset 24  opacity early-out threshold
    uint32_t renderMode = 1u; // offset 28  SdfRenderer::RenderMode
    uint32_t debugFlags = 0u; // offset 32  debug bitfield
    // Water-surface depth clamp (set=1 binding 9 sdfWaterDepth): 1 when the
    // water pass wrote a valid geometry depth this frame. 0 keeps the shader
    // on the solid-depth clamp only.
    float waterDepthEnabled = 0.0f; // offset 36
    // Ray-cast quality: the SDF ray is cast once per raycastPixelSize x
    // raycastPixelSize screen-pixel block (block center), pixelating the SDF
    // output; 1 = one ray per pixel. invScreenSize is the SDF target's
    // inverse size (gl_FragCoord -> UV for the block-center ray).
    alignas(8) glm::vec2 invScreenSize{0.0f}; // offset 40
    float raycastPixelSize = 2.0f;   // offset 48
    // Grass SDF impostor fade band (camera distance in clump scales):
    // below impostorStart the blade LOD only, impostorStart..impostorFull the
    // blades unioned with the fading impostor, at/above impostorFull the
    // impostor only. Exposed in the Impostors widget.
    float impostorStart = 48.0f;     // offset 52
    float impostorFull = 80.0f;      // offset 56
    // Billboard impostor hand-off (Grass Raycast mode): grass clumps beyond
    // this camera distance (metres) are drawn as captured impostor quads by
    // the vegetation impostor pass and skipped by the SDF march. 0 = no
    // hand-off (every clump stays in the march).
    float grassImpostorDistance = 0.0f; // offset 60
    // Distance-tiered SDF LOD (Group C: C3 noise LOD + H7 normal tier).
    // Below sdfLodNear every sample is full detail (tetrahedral normals on
    // hits); beyond sdfLodFar flame deform/spike noise and its wind-lean
    // sample are skipped (no Lipschitz halving there either) and surface
    // normals use forward differences reusing the hit-step dBest. Between
    // the two, deformation stays full while hits already use cheap normals.
    // A value <= 0 disables that tier (the shader substitutes 1e5 = never),
    // so a zero-filled UBO behaves exactly like the pre-LOD march.
    float sdfLodNear = 120.0f; // offset 64
    float sdfLodFar = 360.0f;  // offset 68
    // Phase-B smoke resolve samples (M12 tier, Settings::sdfSmokeSamples,
    // 4..12, default 12 = reference). Shares the std140 tail row with the
    // LOD fields above (offsets 64/68/72 + 4 bytes implicit padding = 80).
    float smokeSamples = 12.0f; // offset 72
};
static_assert(sizeof(SdfParamsUBO) == 80, "SdfParamsUBO must be 80 bytes (std140 padding)");
static_assert(offsetof(SdfParamsUBO, time) == 0, "time offset");
static_assert(offsetof(SdfParamsUBO, maxSteps) == 4, "maxSteps offset");
static_assert(offsetof(SdfParamsUBO, safety) == 8, "safety offset");
static_assert(offsetof(SdfParamsUBO, minStep) == 12, "minStep offset");
static_assert(offsetof(SdfParamsUBO, maxStep) == 16, "maxStep offset");
static_assert(offsetof(SdfParamsUBO, epsilon) == 20, "epsilon offset");
static_assert(offsetof(SdfParamsUBO, earlyTerm) == 24, "earlyTerm offset");
static_assert(offsetof(SdfParamsUBO, renderMode) == 28, "renderMode offset");
static_assert(offsetof(SdfParamsUBO, debugFlags) == 32, "debugFlags offset");
static_assert(offsetof(SdfParamsUBO, waterDepthEnabled) == 36, "waterDepthEnabled offset");
static_assert(offsetof(SdfParamsUBO, invScreenSize) == 40, "invScreenSize offset");
static_assert(offsetof(SdfParamsUBO, raycastPixelSize) == 48, "raycastPixelSize offset");
static_assert(offsetof(SdfParamsUBO, impostorStart) == 52, "impostorStart offset");
static_assert(offsetof(SdfParamsUBO, impostorFull) == 56, "impostorFull offset");
static_assert(offsetof(SdfParamsUBO, grassImpostorDistance) == 60, "grassImpostorDistance offset");
static_assert(offsetof(SdfParamsUBO, sdfLodNear) == 64, "sdfLodNear offset");
static_assert(offsetof(SdfParamsUBO, sdfLodFar) == 68, "sdfLodFar offset");
static_assert(offsetof(SdfParamsUBO, smokeSamples) == 72, "smokeSamples offset");

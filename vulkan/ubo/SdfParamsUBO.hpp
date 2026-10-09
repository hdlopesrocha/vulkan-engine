#pragma once

// Global raymarch / debug parameters (one small UBO, std140, 48 bytes).
// Canonical definition shared with the GLSL twin
// shaders/ubo/SdfParamsUBO.glsl — same names, fields and offsets.
// Independent scalars; renderMode/debugFlags are separate integers (no
// packed float). alignas(16) matches the std140 struct base alignment, so the
// block size is 48 (32 bytes of fields + 16-byte alignment rounding).
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
};
static_assert(sizeof(SdfParamsUBO) == 64, "SdfParamsUBO must be 64 bytes (std140 padding)");
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

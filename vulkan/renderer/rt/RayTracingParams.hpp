#pragma once

#include <cstddef>
#include <cstdint>

#include <glm/glm.hpp>

// Shared ray-tracing parameters UBO (std140, 288 B). Single definition shared
// with the GLSL twin shaders/ubo/RayTracingParams.glsl — same names, fields
// and offsets; streamed per frame via mapped memcpy (handle stable).
// Flags are uint32 (GLSL bool is not a host-shareable block type).
// Member alignment is pinned with alignas to the std140 vector rules
// (vec2 -> 8, vec3/vec4/mat4 -> 16) so the layout never depends on glm packing.
struct alignas(16) RayTracingParams {
    glm::mat4 invViewProj{1.0f};        // offset   0
    glm::mat4 prevViewProj{1.0f};       // offset  64  previous frame's view-projection (temporal SSR)
    alignas(16) glm::vec3 viewPosition{0.0f}; // offset 128
    float maxReflectDistance = 500.0f;  // offset 140
    alignas(16) glm::vec3 sunDirection{0.0f, -1.0f, 0.0f}; // offset 144  xyz = direction TO sun
    float maxRefractDistance = 300.0f;  // offset 156
    alignas(16) glm::vec3 sunColor{1.0f, 1.0f, 0.9f}; // offset 160
    float maxShadowDistance = 12.0f;    // offset 172
    alignas(16) glm::vec3 absorptionColor{0.35f, 0.12f, 0.08f}; // offset 176  Beer-Lambert coeff
    float absorptionScale = 1.0f;       // offset 188
    alignas(8) glm::vec2 rtResolution{0.0f};    // offset 192  dispatch size
    alignas(8) glm::vec2 invRtResolution{0.0f}; // offset 200  1/size
    float roughnessThreshold = 0.6f;    // offset 208
    float waterIor = 1.333f;            // offset 212
    float maxWaterThickness = 6.0f;     // offset 216
    float coarseBoxSize = 0.0f;         // offset 220
    int32_t maxReflectionBounces = 1;   // offset 224
    int32_t debugMode = 0;              // offset 228  see debug_modes.glsl
    float selfSkipDist = 0.0f;          // offset 232
    float reflectionContribMin = 0.02f; // offset 236
    float nearPlane = 0.1f;             // offset 240
    float farPlane = 8092.0f;           // offset 244
    uint32_t reflectionsEnabled = 1u;   // offset 248
    uint32_t refractionsEnabled = 1u;   // offset 252
    uint32_t thicknessEnabled = 1u;     // offset 256
    uint32_t localShadowsEnabled = 0u;  // offset 260
    uint32_t tlasReady = 0u;            // offset 264
    uint32_t useWaterPipeline = 1u;     // offset 268
    uint32_t checkerboardReflections = 1u; // offset 272
    uint32_t singleRay = 1u;            // offset 276
    uint32_t waterReflections = 1u;     // offset 280
    uint32_t rayTracedWaterDepth = 0u;  // offset 284
    // 288 = struct end (multiple of 16; no tail padding needed).
};
static_assert(sizeof(RayTracingParams) == 288, "RayTracingParams must be 288 bytes");
static_assert(offsetof(RayTracingParams, invViewProj) == 0, "invViewProj offset");
static_assert(offsetof(RayTracingParams, prevViewProj) == 64, "prevViewProj offset");
static_assert(offsetof(RayTracingParams, viewPosition) == 128, "viewPosition offset");
static_assert(offsetof(RayTracingParams, maxReflectDistance) == 140, "maxReflectDistance offset");
static_assert(offsetof(RayTracingParams, sunDirection) == 144, "sunDirection offset");
static_assert(offsetof(RayTracingParams, maxRefractDistance) == 156, "maxRefractDistance offset");
static_assert(offsetof(RayTracingParams, sunColor) == 160, "sunColor offset");
static_assert(offsetof(RayTracingParams, maxShadowDistance) == 172, "maxShadowDistance offset");
static_assert(offsetof(RayTracingParams, absorptionColor) == 176, "absorptionColor offset");
static_assert(offsetof(RayTracingParams, absorptionScale) == 188, "absorptionScale offset");
static_assert(offsetof(RayTracingParams, rtResolution) == 192, "rtResolution offset");
static_assert(offsetof(RayTracingParams, invRtResolution) == 200, "invRtResolution offset");
static_assert(offsetof(RayTracingParams, roughnessThreshold) == 208, "roughnessThreshold offset");
static_assert(offsetof(RayTracingParams, waterIor) == 212, "waterIor offset");
static_assert(offsetof(RayTracingParams, maxWaterThickness) == 216, "maxWaterThickness offset");
static_assert(offsetof(RayTracingParams, coarseBoxSize) == 220, "coarseBoxSize offset");
static_assert(offsetof(RayTracingParams, maxReflectionBounces) == 224, "maxReflectionBounces offset");
static_assert(offsetof(RayTracingParams, debugMode) == 228, "debugMode offset");
static_assert(offsetof(RayTracingParams, selfSkipDist) == 232, "selfSkipDist offset");
static_assert(offsetof(RayTracingParams, reflectionContribMin) == 236, "reflectionContribMin offset");
static_assert(offsetof(RayTracingParams, nearPlane) == 240, "nearPlane offset");
static_assert(offsetof(RayTracingParams, farPlane) == 244, "farPlane offset");
static_assert(offsetof(RayTracingParams, reflectionsEnabled) == 248, "reflectionsEnabled offset");
static_assert(offsetof(RayTracingParams, refractionsEnabled) == 252, "refractionsEnabled offset");
static_assert(offsetof(RayTracingParams, thicknessEnabled) == 256, "thicknessEnabled offset");
static_assert(offsetof(RayTracingParams, localShadowsEnabled) == 260, "localShadowsEnabled offset");
static_assert(offsetof(RayTracingParams, tlasReady) == 264, "tlasReady offset");
static_assert(offsetof(RayTracingParams, useWaterPipeline) == 268, "useWaterPipeline offset");
static_assert(offsetof(RayTracingParams, checkerboardReflections) == 272, "checkerboardReflections offset");
static_assert(offsetof(RayTracingParams, singleRay) == 276, "singleRay offset");
static_assert(offsetof(RayTracingParams, waterReflections) == 280, "waterReflections offset");
static_assert(offsetof(RayTracingParams, rayTracedWaterDepth) == 284, "rayTracedWaterDepth offset");

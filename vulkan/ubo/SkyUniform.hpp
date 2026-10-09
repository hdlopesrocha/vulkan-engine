#pragma once
#include <cstddef>
#include <cstdint>

#include <glm/glm.hpp>

// Canonical sky + volumetric-cloud UBO (std140, 224 B), set=0 binding=6.
// Single definition shared with the GLSL twin shaders/ubo/SkyUniform.glsl —
// same names, fields and offsets; uploaded verbatim (memcpy) by SkySphere.
// Colours keep their conceptual vec3 grouping; gradient/cloud scalars are
// individual fields; flags are uint32 (GLSL bool is not a host-shareable
// block type). Member alignment is pinned with alignas to the std140 rules
// (vec3 -> 16) so the layout never depends on glm packing.
struct alignas(16) SkyUniform {
    alignas(16) glm::vec3 horizonColor{0.0f};      // offset   0
    float warmth = 0.0f;                           // offset  12
    alignas(16) glm::vec3 zenithColor{0.0f};       // offset  16
    float exponent = 0.0f;                         // offset  28
    alignas(16) glm::vec3 nightHorizonColor{0.0f}; // offset  32
    float sunFlare = 0.0f;                         // offset  44
    alignas(16) glm::vec3 nightZenithColor{0.0f};  // offset  48
    float nightIntensity = 0.0f;                   // offset  60
    float starIntensity = 0.0f;                    // offset  64
    uint32_t skyMode = 0u;                         // offset  68  0 gradient, 1 grid
    uint32_t cloudsEnabled = 0u;                   // offset  72
    uint32_t lowEnabled = 0u;                      // offset  76
    uint32_t midEnabled = 0u;                      // offset  80
    uint32_t highEnabled = 0u;                     // offset  84
    float densityScale = 0.0f;                     // offset  88
    float windResponse = 0.0f;                     // offset  92  cloud gain on the shared wind field
    float windTime = 0.0f;                         // offset  96  real shared-wind clock (seconds)
    float detailStrength = 0.0f;                   // offset 100
    float cloudTime = 0.0f;                        // offset 104  advanced on CPU
    float shadowStrength = 0.0f;                   // offset 108
    float raymarchSteps = 0.0f;                    // offset 112
    float lightSteps = 0.0f;                       // offset 116
    float lowCoverage = 0.0f;                      // offset 120
    float lowDensity = 0.0f;                       // offset 124
    float lowScale = 0.0f;                         // offset 128
    float lowWindSpeedMul = 0.0f;                  // offset 132
    float lowBaseHeight = 0.0f;                    // offset 136
    float lowThickness = 0.0f;                     // offset 140
    float midCoverage = 0.0f;                      // offset 144
    float midDensity = 0.0f;                       // offset 148
    float midScale = 0.0f;                         // offset 152
    float midWindSpeedMul = 0.0f;                  // offset 156
    float midBaseHeight = 0.0f;                    // offset 160
    float midThickness = 0.0f;                     // offset 164
    float highCoverage = 0.0f;                     // offset 168
    float highDensity = 0.0f;                      // offset 172
    float highScale = 0.0f;                        // offset 176
    float highWindSpeedMul = 0.0f;                 // offset 180
    float highBaseHeight = 0.0f;                   // offset 184
    float highThickness = 0.0f;                    // offset 188
    float silverLining = 0.0f;                     // offset 192
    float ambientBoost = 0.0f;                     // offset 196
    float sunForwardG = 0.0f;                      // offset 200
    float exposure = 0.0f;                         // offset 204
    float timeScale = 0.0f;                        // offset 208
    // Cloud ray-cast quality (Settings::raycastPixelSize): one cloud ray per
    // NxN screen-pixel block (block center). invScreenSize maps gl_FragCoord
    // -> UV and is set per frame from the swapchain size.
    float cloudRaycastPixelSize = 2.0f;            // offset 212
    alignas(8) glm::vec2 invScreenSize{0.0f};      // offset 216
};
static_assert(sizeof(SkyUniform) == 224, "SkyUniform must be 224 bytes (std140 padding)");
static_assert(offsetof(SkyUniform, horizonColor) == 0, "horizonColor offset");
static_assert(offsetof(SkyUniform, warmth) == 12, "warmth offset");
static_assert(offsetof(SkyUniform, zenithColor) == 16, "zenithColor offset");
static_assert(offsetof(SkyUniform, exponent) == 28, "exponent offset");
static_assert(offsetof(SkyUniform, nightHorizonColor) == 32, "nightHorizonColor offset");
static_assert(offsetof(SkyUniform, sunFlare) == 44, "sunFlare offset");
static_assert(offsetof(SkyUniform, nightZenithColor) == 48, "nightZenithColor offset");
static_assert(offsetof(SkyUniform, nightIntensity) == 60, "nightIntensity offset");
static_assert(offsetof(SkyUniform, starIntensity) == 64, "starIntensity offset");
static_assert(offsetof(SkyUniform, skyMode) == 68, "skyMode offset");
static_assert(offsetof(SkyUniform, cloudsEnabled) == 72, "cloudsEnabled offset");
static_assert(offsetof(SkyUniform, lowEnabled) == 76, "lowEnabled offset");
static_assert(offsetof(SkyUniform, midEnabled) == 80, "midEnabled offset");
static_assert(offsetof(SkyUniform, highEnabled) == 84, "highEnabled offset");
static_assert(offsetof(SkyUniform, densityScale) == 88, "densityScale offset");
static_assert(offsetof(SkyUniform, windResponse) == 92, "windResponse offset");
static_assert(offsetof(SkyUniform, windTime) == 96, "windTime offset");
static_assert(offsetof(SkyUniform, detailStrength) == 100, "detailStrength offset");
static_assert(offsetof(SkyUniform, cloudTime) == 104, "cloudTime offset");
static_assert(offsetof(SkyUniform, shadowStrength) == 108, "shadowStrength offset");
static_assert(offsetof(SkyUniform, raymarchSteps) == 112, "raymarchSteps offset");
static_assert(offsetof(SkyUniform, lightSteps) == 116, "lightSteps offset");
static_assert(offsetof(SkyUniform, lowCoverage) == 120, "lowCoverage offset");
static_assert(offsetof(SkyUniform, lowDensity) == 124, "lowDensity offset");
static_assert(offsetof(SkyUniform, lowScale) == 128, "lowScale offset");
static_assert(offsetof(SkyUniform, lowWindSpeedMul) == 132, "lowWindSpeedMul offset");
static_assert(offsetof(SkyUniform, lowBaseHeight) == 136, "lowBaseHeight offset");
static_assert(offsetof(SkyUniform, lowThickness) == 140, "lowThickness offset");
static_assert(offsetof(SkyUniform, midCoverage) == 144, "midCoverage offset");
static_assert(offsetof(SkyUniform, midDensity) == 148, "midDensity offset");
static_assert(offsetof(SkyUniform, midScale) == 152, "midScale offset");
static_assert(offsetof(SkyUniform, midWindSpeedMul) == 156, "midWindSpeedMul offset");
static_assert(offsetof(SkyUniform, midBaseHeight) == 160, "midBaseHeight offset");
static_assert(offsetof(SkyUniform, midThickness) == 164, "midThickness offset");
static_assert(offsetof(SkyUniform, highCoverage) == 168, "highCoverage offset");
static_assert(offsetof(SkyUniform, highDensity) == 172, "highDensity offset");
static_assert(offsetof(SkyUniform, highScale) == 176, "highScale offset");
static_assert(offsetof(SkyUniform, highWindSpeedMul) == 180, "highWindSpeedMul offset");
static_assert(offsetof(SkyUniform, highBaseHeight) == 184, "highBaseHeight offset");
static_assert(offsetof(SkyUniform, highThickness) == 188, "highThickness offset");
static_assert(offsetof(SkyUniform, silverLining) == 192, "silverLining offset");
static_assert(offsetof(SkyUniform, ambientBoost) == 196, "ambientBoost offset");
static_assert(offsetof(SkyUniform, sunForwardG) == 200, "sunForwardG offset");
static_assert(offsetof(SkyUniform, exposure) == 204, "exposure offset");
static_assert(offsetof(SkyUniform, timeScale) == 208, "timeScale offset");
static_assert(offsetof(SkyUniform, cloudRaycastPixelSize) == 212, "cloudRaycastPixelSize offset");
static_assert(offsetof(SkyUniform, invScreenSize) == 216, "invScreenSize offset");

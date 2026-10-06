#pragma once
#include <cstddef>

#include <glm/glm.hpp>

// Canonical water composite/frame UBO (std140, 176 B). Single definition
// shared with the GLSL twin shaders/ubo/WaterUBO.glsl — same names, fields
// and offsets; uploaded verbatim (memcpy) by PostProcessRenderer.
// Member alignment is pinned with alignas to the std140 vector rules
// (vec2 -> 8, vec3/vec4/mat4 -> 16) so the layout never depends on glm
// packing.
struct alignas(16) WaterUBO {
    glm::mat4 viewProjection{1.0f};    // offset  0
    glm::mat4 invViewProjection{1.0f}; // offset 64
    alignas(16) glm::vec3 viewPosition{0.0f}; // offset 128  camera world position
    float brushAlpha = 0.0f;           // offset 140  brush overlay opacity (0-1)
    alignas(8) glm::vec2 screenSize{0.0f};    // offset 144  (width, height)
    alignas(8) glm::vec2 invScreenSize{0.0f}; // offset 152  (1/width, 1/height)
    float brushMode = 0.0f;            // offset 160  0=overlay, 2=PAINT
    // H4: 1 when at least one water layer needs the final-pass blur, i.e. the
    // water pass wrote the body/column aux attachments; 0 lets the composite
    // skip both fetches entirely.
    float waterBlurEnabled = 0.0f;     // offset 164
    // M12 (perf report 22): 1 when the vegetation offscreen targets are
    // downscaled; the composite then takes the closest of the 2x2 veg-depth
    // taps instead of one bilinear sample.
    float vegetationScaled = 0.0f;     // offset 168
    // offset 172..176: std140 block size rounding (multiple of 16).
};
static_assert(sizeof(WaterUBO) == 176, "WaterUBO must be 176 bytes (std140 padding)");
static_assert(offsetof(WaterUBO, viewProjection) == 0, "viewProjection offset");
static_assert(offsetof(WaterUBO, invViewProjection) == 64, "invViewProjection offset");
static_assert(offsetof(WaterUBO, viewPosition) == 128, "viewPosition offset");
static_assert(offsetof(WaterUBO, brushAlpha) == 140, "brushAlpha offset");
static_assert(offsetof(WaterUBO, screenSize) == 144, "screenSize offset");
static_assert(offsetof(WaterUBO, invScreenSize) == 152, "invScreenSize offset");
static_assert(offsetof(WaterUBO, brushMode) == 160, "brushMode offset");
static_assert(offsetof(WaterUBO, waterBlurEnabled) == 164, "waterBlurEnabled offset");
static_assert(offsetof(WaterUBO, vegetationScaled) == 168, "vegetationScaled offset");

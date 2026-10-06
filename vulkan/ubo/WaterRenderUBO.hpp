#pragma once
#include <cstddef>
#include <cstdint>

// Canonical water render parameters UBO (std140, 32 B). Single definition
// shared with the GLSL twin shaders/ubo/WaterRenderUBO.glsl — same names,
// fields and offsets; uploaded verbatim (memcpy) by WaterRenderer.
// Booleans are uint32 flags (GLSL bool is not a host-shareable block type).
struct alignas(16) WaterRenderUBO {
    float waterTime = 0.0f;            // offset  0  wave/anim clock (s)
    uint32_t refractionAllowed = 0u;   // offset  4  1 = refraction lobe enabled
    uint32_t reflectionAllowed = 0u;   // offset  8  1 = reflection lobe enabled
    uint32_t blurAllowed = 0u;         // offset 12  1 = blur enabled (Settings toggle)
    uint32_t solidDepthIsCurrent = 0u; // offset 16  1 = binding 7 holds THIS frame's
                                       //            solid depth (gates solid occlusion)
    // offset 20..32: std140 block size rounding (multiple of 16); no functional fields.
};
static_assert(sizeof(WaterRenderUBO) == 32, "WaterRenderUBO must be 32 bytes (std140 padding)");
static_assert(offsetof(WaterRenderUBO, waterTime) == 0, "waterTime offset");
static_assert(offsetof(WaterRenderUBO, refractionAllowed) == 4, "refractionAllowed offset");
static_assert(offsetof(WaterRenderUBO, reflectionAllowed) == 8, "reflectionAllowed offset");
static_assert(offsetof(WaterRenderUBO, blurAllowed) == 12, "blurAllowed offset");
static_assert(offsetof(WaterRenderUBO, solidDepthIsCurrent) == 16, "solidDepthIsCurrent offset");

#pragma once
#include <cstddef>

#include <glm/glm.hpp>

// Canonical impostor-capture camera UBO (std140, 112 B). Single definition
// shared with the GLSL twin shaders/ubo/CaptureUBO.glsl — same names, fields
// and offsets; uploaded verbatim into per-view dynamic-offset slots.
// Member alignment is pinned with alignas to the std140 vector rules
// (vec3/vec4/mat4 -> 16) so the layout never depends on glm packing.
struct alignas(16) CaptureUBO {
    glm::mat4 viewProjection{1.0f};          // offset  0
    alignas(16) glm::vec3 viewPosition{0.0f}; // offset 64  capture eye position
    alignas(16) glm::vec3 lightDirection{0.0f}; // offset 80
    alignas(16) glm::vec3 lightColor{0.0f};  // offset 96
    // offset 108..112: std140 block size rounding (multiple of 16).
};
static_assert(sizeof(CaptureUBO) == 112, "CaptureUBO must be 112 bytes (std140 padding)");
static_assert(offsetof(CaptureUBO, viewProjection) == 0, "viewProjection offset");
static_assert(offsetof(CaptureUBO, viewPosition) == 64, "viewPosition offset");
static_assert(offsetof(CaptureUBO, lightDirection) == 80, "lightDirection offset");
static_assert(offsetof(CaptureUBO, lightColor) == 96, "lightColor offset");

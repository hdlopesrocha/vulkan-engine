#pragma once

// Canonical SDF instance — ONE definition shared by the CPU scene builder and
// the GPU shaders (GLSL twin: shaders/types/SdfInstance.glsl, identical
// layout). Uploaded verbatim as an array into the set=1 binding=0 SSBO
// (std430; array stride = sizeof).
//
// position/rotation are conceptual vectors (SdfModel transform);
// scale/heightScale/radiusScale/intensity/seed and the palette indices are
// independent scalars. boundsMin/boundsMax are the precomputed (rotated)
// world AABB used by the shader's bound test and the CPU grid build, so the
// same struct can be uploaded without any conversion.
//
// Member alignment is pinned with alignas to the std430 vector rules
// (vec2 -> 8, vec3/vec4 -> 16) so the layout never depends on glm packing.
#include <cstddef>
#include <cstdint>

#include <glm/glm.hpp>

struct SdfInstance {
    alignas(16) glm::vec3 position{0.0f};   // offset  0  world position
    float scale = 1.0f;                     // offset 12  uniform scale
    alignas(16) glm::vec3 rotation{0.0f};   // offset 16  euler radians, R = Rx*Ry*Rz (SdfModel)
    float heightScale = 1.0f;               // offset 28  per-instance vertical stretch
    float radiusScale = 1.0f;               // offset 32
    float intensity = 1.0f;                 // offset 36
    float seed = 0.0f;                      // offset 40  animation seed
    uint32_t defIdx = 0u;                   // offset 44  definition index
    uint32_t matIdx = 0u;                   // offset 48  material index
    uint32_t containerIdx = 0u;             // offset 52  owning container (CPU grid build)
    // offset 56..64 is padding: boundsMin must start at a 16-byte boundary.
    alignas(16) glm::vec3 boundsMin{0.0f};  // offset 64  world AABB min
    // offset 76..80 is padding: boundsMax must start at a 16-byte boundary.
    alignas(16) glm::vec3 boundsMax{0.0f};  // offset 80  world AABB max
    // offset 92..96 is padding: the struct size must be a multiple of 16.
};
static_assert(sizeof(SdfInstance) == 96, "SdfInstance must be 96 bytes");
static_assert(offsetof(SdfInstance, position) == 0, "position offset");
static_assert(offsetof(SdfInstance, scale) == 12, "scale offset");
static_assert(offsetof(SdfInstance, rotation) == 16, "rotation offset");
static_assert(offsetof(SdfInstance, heightScale) == 28, "heightScale offset");
static_assert(offsetof(SdfInstance, radiusScale) == 32, "radiusScale offset");
static_assert(offsetof(SdfInstance, intensity) == 36, "intensity offset");
static_assert(offsetof(SdfInstance, seed) == 40, "seed offset");
static_assert(offsetof(SdfInstance, defIdx) == 44, "defIdx offset");
static_assert(offsetof(SdfInstance, matIdx) == 48, "matIdx offset");
static_assert(offsetof(SdfInstance, containerIdx) == 52, "containerIdx offset");
static_assert(offsetof(SdfInstance, boundsMin) == 64, "boundsMin offset");
static_assert(offsetof(SdfInstance, boundsMax) == 80, "boundsMax offset");

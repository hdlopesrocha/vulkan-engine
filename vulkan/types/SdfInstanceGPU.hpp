#pragma once

// Per-instance transform + palette indices + precomputed world AABB
// (std430, 96 bytes, vec4-aligned).
#include <glm/glm.hpp>
#include <cstdint>

struct SdfInstanceGPU {
    glm::vec4 posScale{0.0f};   // xyz = world position, w = uniform scale
    glm::vec4 rotSeed{0.0f};    // xyz = euler angles (radians, XYZ order), w = animation seed
    glm::vec4 sizeParams{0.0f}; // x = height scale, y = radius scale, z = intensity, w = unused
    glm::uvec4 indices{0u};     // x = definition index, y = material index, z = container index, w = flags
    glm::vec4 boundsMin{0.0f};  // world AABB min, w unused
    glm::vec4 boundsMax{0.0f};  // world AABB max, w unused
};
static_assert(sizeof(SdfInstanceGPU) == 96, "SdfInstanceGPU must be 96 bytes");
static_assert(sizeof(SdfInstanceGPU) % 16 == 0, "SdfInstanceGPU must be multiple of 16");

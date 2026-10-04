#pragma once

// Spatial container (broadphase volume) with an embedded uniform grid
// (std430, 64 bytes, vec4-aligned).
#include <glm/glm.hpp>
#include <cstdint>

struct SdfContainerGPU {
    glm::vec4 boundsMin{0.0f};
    glm::vec4 boundsMax{0.0f};
    glm::uvec4 gridInfo{0u};   // x,y,z = resolution, w = instanceStart (offset into global index list)
    glm::uvec4 gridOffset{0u}; // x = cell buffer offset, y = index buffer offset, z = instance count, w = flags
};
static_assert(sizeof(SdfContainerGPU) == 64, "SdfContainerGPU must be 64 bytes");
static_assert(sizeof(SdfContainerGPU) % 16 == 0, "SdfContainerGPU must be multiple of 16");

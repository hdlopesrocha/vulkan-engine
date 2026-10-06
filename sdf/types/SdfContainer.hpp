#pragma once

// Canonical SDF container — ONE definition shared by the CPU scene builder
// and the GPU shaders (GLSL twin: shaders/types/SdfContainer.glsl,
// identical layout). Uploaded verbatim as an array into the set=1 binding=3
// SSBO (std430; array stride = sizeof).
//
// boundsMin/boundsMax are conceptual vectors (the broadphase AABB);
// resX/resY/resZ (uniform-grid resolution) and cellStart (global offset into
// the cell buffer) are independent scalars.
//
// The per-container instance membership is CPU-only bookkeeping and does NOT
// live in this GPU struct.
//
// Member alignment is pinned with alignas to the std430 vector rules
// (vec3 -> 16) so the layout never depends on glm packing.
#include <cstddef>
#include <cstdint>

#include <glm/glm.hpp>

struct SdfContainer {
    alignas(16) glm::vec3 boundsMin{0.0f}; // offset  0  world AABB min
    alignas(16) glm::vec3 boundsMax{0.0f}; // offset 16  world AABB max
    uint32_t resX = 8u;                    // offset 28  grid resolution X (> 0)
    uint32_t resY = 8u;                    // offset 32
    uint32_t resZ = 8u;                    // offset 36
    uint32_t cellStart = 0u;               // offset 40  global cell-buffer start
};
static_assert(sizeof(SdfContainer) == 48, "SdfContainer must be 48 bytes");
static_assert(offsetof(SdfContainer, boundsMin) == 0, "boundsMin offset");
static_assert(offsetof(SdfContainer, boundsMax) == 16, "boundsMax offset");
static_assert(offsetof(SdfContainer, resX) == 28, "resX offset");
static_assert(offsetof(SdfContainer, resY) == 32, "resY offset");
static_assert(offsetof(SdfContainer, resZ) == 36, "resZ offset");
static_assert(offsetof(SdfContainer, cellStart) == 40, "cellStart offset");

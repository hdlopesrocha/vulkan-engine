#pragma once

// Canonical SDF uniform-grid cell — ONE definition shared by the CPU scene
// builder and the GPU shaders (GLSL twin: shaders/types/SdfGridCell.glsl,
// identical layout). Uploaded verbatim as an array into the set=1 binding=4
// SSBO (std430; array stride = sizeof == 8; no padding: two adjacent
// scalars satisfy every alignment rule).
#include <cstddef>
#include <cstdint>

struct SdfGridCell {
    uint32_t offset = 0u; // offset 0  first index in the global index buffer
    uint32_t count = 0u;  // offset 4  index count
};
static_assert(sizeof(SdfGridCell) == 8, "SdfGridCell must be 8 bytes");
static_assert(offsetof(SdfGridCell, offset) == 0, "offset member offset");
static_assert(offsetof(SdfGridCell, count) == 4, "count member offset");

#pragma once

// One uniform-grid cell: contiguous range [offset, offset+count) into the
// container's (or global) instance-index list (std430, 16 bytes).
#include <cstdint>

struct SdfGridCellGPU {
    uint32_t offset = 0;
    uint32_t count = 0;
    uint32_t _pad0 = 0;
    uint32_t _pad1 = 0;
};
static_assert(sizeof(SdfGridCellGPU) == 16, "SdfGridCellGPU must be 16 bytes");

#pragma once

// Plain CPU-side uniform-grid topology for one SDF container: cells reference
// contiguous ranges [offset, offset+count) into indices. No Vulkan dependency;
// the Vulkan-side adapter (vulkan/renderer/SdfSceneFlatten.*) widens cells to
// SdfGridCellGPU at flatten time.
#include <cstdint>
#include <vector>

namespace sdf_gpu {

struct SdfUniformGrid {
    struct Cell {
        uint32_t offset = 0;
        uint32_t count = 0;
    };
    std::vector<Cell> cells;
    std::vector<uint32_t> indices;
};

} // namespace sdf_gpu

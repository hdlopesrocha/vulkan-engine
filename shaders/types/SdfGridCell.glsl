#ifndef SDF_GRID_CELL_GLSL
#define SDF_GRID_CELL_GLSL

// Canonical SDF uniform-grid cell (std430, 8 B, no padding). Single
// definition shared with the CPU scene builder:
// sdf/types/SdfGridCell.hpp — identical layout and offsets.
struct SdfGridCell {
    uint offset; // offset 0  first index in the global index buffer
    uint count;  // offset 4  index count
};

#endif // SDF_GRID_CELL_GLSL

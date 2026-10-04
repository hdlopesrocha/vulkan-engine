#ifndef SDF_GRID_CELL_G_P_U_GLSL
#define SDF_GRID_CELL_G_P_U_GLSL

// Extracted from shaders/includes/sdf_material.glsl (single-struct GLSL type).

struct SdfGridCellGPU {
    uint offset;
    uint count;
    uint _pad0;
    uint _pad1;
};

#endif // SDF_GRID_CELL_G_P_U_GLSL

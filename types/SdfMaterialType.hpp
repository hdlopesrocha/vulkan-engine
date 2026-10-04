#pragma once

// Material shading mode encoded in SdfMaterialGPU::surfaceParams.w (as float)
#include <cstdint>

enum class SdfMaterialType : uint32_t {
    Surface = 0,
    Emissive = 1,
    Volume = 2,
    Transparent = 3
};

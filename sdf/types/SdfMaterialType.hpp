#pragma once

// Material shading mode encoded in SdfMaterial::surfaceParams.w (as float).
// GLSL twin: shaders/types/SdfMaterialType.glsl (SDF_MAT_*).
#include <cstdint>

enum class SdfMaterialType : uint32_t {
    Surface = 0,
    Emissive = 1,
    Volume = 2,
    Transparent = 3
};

#pragma once

// Deform flag bits encoded in SdfDefinition::meta.z
// bit0=noise, bit1=twist, bit2=bend, bit3=taper, bit4=repeat.
// GLSL twin: shaders/types/SdfDeformFlags.glsl (SDF_DEFORM_*).
#include <cstdint>

enum class SdfDeformFlags : uint32_t {
    None = 0u,
    Noise = 1u << 0,
    Twist = 1u << 1,
    Bend = 1u << 2,
    Taper = 1u << 3,
    Repeat = 1u << 4
};

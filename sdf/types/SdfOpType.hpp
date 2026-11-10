#pragma once

// CSG / blending operator encoded in SdfDefinition::meta.y.
// GLSL twin: shaders/types/SdfOpType.glsl (SDF_OP_*).
#include <cstdint>

enum class SdfOpType : uint32_t {
    Union = 0,
    Intersection = 1,
    Subtraction = 2,
    SmoothUnion = 3,
    SmoothIntersection = 4,
    SmoothSubtraction = 5,
    // Linear blend between the accumulated field and the current primitive:
    // mix(accumulated, primitive, smoothK), with smoothK clamped to [0,1].
    Interpolation = 6
};

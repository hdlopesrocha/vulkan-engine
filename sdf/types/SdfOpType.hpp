#pragma once

// CSG / blending operator encoded in SdfDefinitionGPU::meta.y
#include <cstdint>

enum class SdfOpType : uint32_t {
    Union = 0,
    Intersection = 1,
    Subtraction = 2,
    SmoothUnion = 3,
    SmoothIntersection = 4,
    SmoothSubtraction = 5
};

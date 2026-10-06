#ifndef WATER_RENDER_U_B_O_GLSL
#define WATER_RENDER_U_B_O_GLSL

// Canonical water render parameters (std140, 32 B). Single definition shared
// with the CPU: vulkan/ubo/WaterRenderUBO.hpp — same names, fields and
// offsets.
// Offset table (verified against the C++ static_asserts):
//   waterTime 0, refractionAllowed 4, reflectionAllowed 8, blurAllowed 12,
//   solidDepthIsCurrent 16. 20..32 is std140 block-size rounding.
// Booleans are uint flags (compare with != 0u).
struct WaterRenderUBO {
    float waterTime;            // offset  0
    uint refractionAllowed;     // offset  4
    uint reflectionAllowed;     // offset  8
    uint blurAllowed;           // offset 12
    uint solidDepthIsCurrent;   // offset 16
};

#endif // WATER_RENDER_U_B_O_GLSL

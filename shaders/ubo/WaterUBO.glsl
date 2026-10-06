#ifndef WATER_U_B_O_GLSL
#define WATER_U_B_O_GLSL

// Canonical water composite/frame parameters (std140, 176 B). Single
// definition shared with the CPU: vulkan/ubo/WaterUBO.hpp — same names,
// fields and offsets.
// Offset table (verified against the C++ static_asserts):
//   viewProjection 0, invViewProjection 64, viewPosition 128, brushAlpha 140,
//   screenSize 144, invScreenSize 152, brushMode 160, waterBlurEnabled 164,
//   vegetationScaled 168. 172..176 is std140 block-size rounding.
struct WaterUBO {
    mat4 viewProjection;    // offset   0
    mat4 invViewProjection; // offset  64
    vec3 viewPosition;      // offset 128  camera world position
    float brushAlpha;       // offset 140  brush overlay opacity (0-1)
    vec2 screenSize;        // offset 144  (width, height)
    vec2 invScreenSize;     // offset 152  (1/width, 1/height)
    float brushMode;        // offset 160  0=overlay, 2=PAINT
    float waterBlurEnabled; // offset 164
    float vegetationScaled; // offset 168
};

#endif // WATER_U_B_O_GLSL

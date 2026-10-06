#ifndef CAPTURE_U_B_O_GLSL
#define CAPTURE_U_B_O_GLSL

// Canonical impostor-capture camera UBO (std140, 112 B). Single definition
// shared with the CPU: vulkan/ubo/CaptureUBO.hpp — same names, fields and
// offsets.
// Offset table (verified against the C++ static_asserts):
//   viewProjection 0, viewPosition 64, lightDirection 80, lightColor 96.
// 108..112 is std140 block-size rounding.
struct CaptureUBO {
    mat4 viewProjection;  // offset  0
    vec3 viewPosition;    // offset 64  capture eye position
    vec3 lightDirection;  // offset 80
    vec3 lightColor;      // offset 96
};

#endif // CAPTURE_U_B_O_GLSL

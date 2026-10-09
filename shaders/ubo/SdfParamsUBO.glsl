#ifndef SDF_PARAMS_U_B_O_GLSL
#define SDF_PARAMS_U_B_O_GLSL

// Global raymarch / debug parameters (std140, 48 B). Canonical definition
// shared with the CPU: vulkan/ubo/SdfParamsUBO.hpp — same names, fields and
// offsets.
// Offset table (verified against the C++ static_asserts):
//   time 0, maxSteps 4, safety 8, minStep 12, maxStep 16, epsilon 20,
//   earlyTerm 24, renderMode 28, debugFlags 32, waterDepthEnabled 36,
//   invScreenSize 40, raycastPixelSize 48, impostorStart 52, impostorFull 56.
// 60..64 is std140 padding (block size is a multiple of 16).
struct SdfParamsUBO {
    float time;        // offset  0  global SDF time (s)
    float maxSteps;    // offset  4  march step budget
    float safety;      // offset  8  sphere-trace step safety
    float minStep;     // offset 12
    float maxStep;     // offset 16
    float epsilon;     // offset 20  hit epsilon
    float earlyTerm;   // offset 24  opacity early-out threshold
    uint renderMode;   // offset 28  SdfRenderer::RenderMode
    uint debugFlags;   // offset 32  debug bitfield
    float waterDepthEnabled; // offset 36  1 = water depth clamp active (binding 9)
    vec2 invScreenSize;      // offset 40  SDF target inverse size (ray UV)
    float raycastPixelSize;  // offset 48  ray per NxN block (1 = per pixel)
    float impostorStart;     // offset 52  grass impostor fade start (clump scales)
    float impostorFull;      // offset 56  grass impostor fade full (clump scales)
};

#endif // SDF_PARAMS_U_B_O_GLSL

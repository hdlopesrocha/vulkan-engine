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

// Stride invariant (L14, perf report 26): views are consumed through a
// UNIFORM_BUFFER_DYNAMIC descriptor, so the host-side per-view stride MUST
// stay a multiple of minUniformBufferOffsetAlignment (ImpostorCapture::init
// uses the align-up of this 112 B struct). Packing views any tighter would
// break the dynamic-offset alignment rule and is not validation-clean.
// Layout note: the VEG_CAPTURE vertex stage (VegetationRenderer.vert
// -DVEG_CAPTURE=1) declares set 0 as a UniformObject block, whose
// viewPosition sits at offset 320 — outside this 112 B descriptor range. That
// read executes but is benign ONLY because capture forces impostorDistance
// = 0, which dead-branches its sole use (the distance cull); viewProjection
// (offset 0, identical in both layouts) is the only capture VS field that
// resolves in range. Do not read other UniformObject tail fields during
// capture without switching this block to the matching layout
// (vegetation-shader owner). GPU-assisted validation may still report the
// out-of-range tail read — pre-existing, owned by that worker. See
// ImpostorCapture.hpp uboStride.

#endif // CAPTURE_U_B_O_GLSL

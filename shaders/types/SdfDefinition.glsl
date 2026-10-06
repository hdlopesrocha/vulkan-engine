#ifndef SDF_DEFINITION_GLSL
#define SDF_DEFINITION_GLSL

// Canonical SDF primitive definition (std430, 48 B). Single definition shared
// with the CPU scene builder: sdf/types/SdfDefinition.hpp — identical
// layout, names and offsets.
// Offset table (verified against the C++ static_asserts):
//   params0 0, params1 16, prim 32, op 36, deformFlags 40, smoothK 44.
// params0/params1 are the per-primitive parameter blocks (one tagged pack
// each, selected by `prim`); see the C++ header for the full convention.
// prim ids:     shaders/types/SdfPrimitiveType.glsl (SDF_PRIM_*)
// op ids:       shaders/types/SdfOpType.glsl (SDF_OP_*)
// deformFlags:  shaders/types/SdfDeformFlags.glsl (SDF_DEFORM_*)
struct SdfDefinition {
    vec4 params0;      // offset  0  primitive parameter block 1
    vec4 params1;      // offset 16  primitive parameter block 2
    uint prim;         // offset 32
    uint op;           // offset 36
    uint deformFlags;  // offset 40
    float smoothK;     // offset 44  smooth-op blend radius
};

#endif // SDF_DEFINITION_GLSL

#ifndef SDF_DEFORM_FLAGS_GLSL
#define SDF_DEFORM_FLAGS_GLSL

// Deform flag bits encoded in SdfDefinition::meta.z.
// CPU twin: sdf/types/SdfDeformFlags.hpp (SdfDeformFlags).
#define SDF_DEFORM_NOISE (1u << 0)
#define SDF_DEFORM_TWIST (1u << 1)
#define SDF_DEFORM_BEND (1u << 2)
#define SDF_DEFORM_TAPER (1u << 3)
#define SDF_DEFORM_REPEAT (1u << 4)

#endif // SDF_DEFORM_FLAGS_GLSL

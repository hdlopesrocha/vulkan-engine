#ifndef SDF_OP_TYPE_GLSL
#define SDF_OP_TYPE_GLSL

// CSG / blending operator encoded in SdfDefinition::meta.y.
// CPU twin: sdf/types/SdfOpType.hpp (SdfOpType).
#define SDF_OP_UNION 0u
#define SDF_OP_INTERSECTION 1u
#define SDF_OP_SUBTRACTION 2u
#define SDF_OP_SMOOTH_UNION 3u
#define SDF_OP_SMOOTH_INTERSECTION 4u
#define SDF_OP_SMOOTH_SUBTRACTION 5u

#endif // SDF_OP_TYPE_GLSL

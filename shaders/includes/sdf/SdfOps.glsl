// Generic SDF combinators and space transforms (IQ-style exact/polynomial).
// Implementations live one file per operation under sdf/operations/; this
// file stays as the single umbrella include used by the generic SDF renderer,
// the cloud shader and the generic SdfModel.
//
// No textures, no vendor-specific behavior.

#ifndef SDF_OPS_GLSL
#define SDF_OPS_GLSL

#include "operations/Combine.glsl"
#include "operations/Repeat.glsl"
#include "operations/Mirror.glsl"
#include "operations/EulerMat.glsl"

#endif // SDF_OPS_GLSL

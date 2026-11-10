// Generic SDF primitive library (exact IQ-style formulas).
// Implementations live one file per primitive under sdf/primitives/; this
// file stays as the single umbrella include used by the generic SDF renderer
// and the cloud shader.
//
// All primitives share one signature so the dispatcher can evaluate any
// definition: float sdfX(vec3 p, vec4 p0, vec4 p1).
//
// Parameter packing (matches sdf/types/SdfDefinition.hpp + sdf/types/SdfScene.cpp):
//   sphere:      p0.x = radius
//   box:         p0.xyz = half extents
//   rounded box: p0.xyz = half extents (outer), p1.x = corner radius
//   capsule:     Y-aligned: p0.x = radius, p0.y = half height of the
//                cylindrical section (total height = 2*halfHeight + 2*radius)
//   cylinder:    p0.x = radius, p0.y = half height (Y axis, centered)
//   cone:        p0.x = base radius, p0.y = full height (base at -h/2, apex +h/2)
//   torus:       p0.xy = vec2(R, r) (major, minor radii, around Y)
//   plane:       p0.xyz = normal (normalized internally), p0.w = offset
//                (signed distance = dot(p, n) - offset)
//   flame:       p0.x = base radius, p0.y = height, p1.x = tip radius
//   smoke:       p0.x = maximum radius (growth applied by smokeMarchSDF)
// Unused components are ignored. No textures, no vendor-specific behavior.

#ifndef SDF_PRIMITIVES_GLSL
#define SDF_PRIMITIVES_GLSL

#include "primitives/Dispatcher.glsl"

#endif // SDF_PRIMITIVES_GLSL

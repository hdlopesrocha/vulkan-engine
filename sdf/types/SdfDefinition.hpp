#pragma once

// Canonical SDF primitive definition — ONE definition shared by the CPU
// scene builder and the GPU shaders (GLSL twin:
// shaders/types/SdfDefinition.glsl, identical layout). Uploaded verbatim
// as an array into the set=1 binding=1 SSBO (std430; array stride = sizeof).
//
// Member alignment is pinned with alignas to the std430 vector rules
// (vec3/vec4 -> 16) so the layout never depends on glm's default packing.
//
// params0/params1 are the per-primitive parameter blocks (one conceptual
// tagged parameter pack each, selected by `prim`):
//   Sphere:     params0.x = radius
//   Box:        params0.xyz = half extents
//   RoundedBox: params0.xyz = half extents, params1.x = corner radius
//   Capsule (Y-aligned): params0.x = radius, params0.y = half height
//   Cylinder (Y-aligned): params0.x = radius, params0.y = half height
//   Cone (Y-aligned, centered): params0.x = base radius, params0.y = height
//   Torus (XZ plane): params0.x = major radius R, params0.y = minor radius r
//   Plane:      params0.xyz = normal, params0.w = offset
//   Flame:      params0.x = base radius, params0.y = height; params1.x = tip
//               radius, params1.y = spikiness, params1.z = spike frequency
//   Smoke:      params0.x = maximum radius, params0.y = seed
//   Grass:      params0.x = clump radius, params0.y = blade height,
//               params0.z = blade width, params0.w = blade count;
//               params1.x = curvature (tip offset fraction of height),
//               params1.y = maximum wind lean (radians), params1.z = wind
//               gain (lean radians per m/s), params1.w = tip width fraction
//
// prim/op/deformFlags/smoothK are independent scalars (no artificial vec4).
#include <cstddef>
#include <cstdint>

#include <glm/glm.hpp>

#include "sdf/types/SdfPrimitiveType.hpp"
#include "sdf/types/SdfOpType.hpp"

struct SdfDefinition {
    alignas(16) glm::vec4 params0{0.0f};              // offset  0
    alignas(16) glm::vec4 params1{0.0f};              // offset 16
    SdfPrimitiveType prim = SdfPrimitiveType::Sphere; // offset 32
    SdfOpType op = SdfOpType::Union;                  // offset 36
    uint32_t deformFlags = 0u;                        // offset 40 (SdfDeformFlags bits)
    float smoothK = 0.5f;                             // offset 44 (smooth-op blend radius)
};
static_assert(sizeof(SdfDefinition) == 48, "SdfDefinition must be 48 bytes");
static_assert(offsetof(SdfDefinition, params0) == 0, "params0 offset");
static_assert(offsetof(SdfDefinition, params1) == 16, "params1 offset");
static_assert(offsetof(SdfDefinition, prim) == 32, "prim offset");
static_assert(offsetof(SdfDefinition, op) == 36, "op offset");
static_assert(offsetof(SdfDefinition, deformFlags) == 40, "deformFlags offset");
static_assert(offsetof(SdfDefinition, smoothK) == 44, "smoothK offset");

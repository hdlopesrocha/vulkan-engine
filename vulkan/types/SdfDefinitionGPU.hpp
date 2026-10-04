#pragma once

// Single SDF primitive definition (std430, 48 bytes, vec4-aligned).
// params0/params1 convention (local space, before instance scale):
//   Sphere:     params0.x = radius
//   Box:        params0.xyz = half extents
//   RoundedBox: params0.xyz = half extents, params1.x = corner radius
//   Capsule (Y-aligned): params0.x = radius, params0.y = half height of
//               cylindrical section (total height = 2*halfHeight + 2*radius)
//   Cylinder (Y-aligned): params0.x = radius, params0.y = half height
//   Cone (Y-aligned, centered): params0.x = base radius, params0.y = full height
//   Torus (in XZ plane): params0.x = major radius R, params0.y = minor radius r
//   Plane:      params0.xyz = normal, params0.w = offset (signed distance
//               convention: dot(n,p)+offset); bounds use a large thin box.
//   Flame:      params0.x = base radius, params0.y = flame height (base disk
//               at local y=0, tip at y=height); params1.x = tip radius
//               (0 = sharp cone tip, = base radius = capsule), params1.y =
//               spikiness amplitude (0 = smooth rounded capsule), params1.z =
//               spike frequency (tongue count), params1.w unused.
//   Smoke:      params0.x = maximum radius (marching bounds + loop anchor),
//               params0.y = per-instance seed, params0.zw unused. Growth,
//               dissipation, noise, wind, bullets and render tuning live in
//               the smoke state buffer (set=1 binding 8, SmokeFragBulletGPU below),
//               NOT here, so widget tweaks never rebuild scene geometry.
// meta.x = SdfPrimitiveType, meta.y = SdfOpType,
// meta.z = deform flags (SdfDeformFlags bits),
// meta.w = extra (bit-cast float smoothK for smooth ops).
#include <glm/glm.hpp>
#include <cstdint>

struct SdfDefinitionGPU {
    glm::vec4 params0{0.0f};
    glm::vec4 params1{0.0f};
    glm::uvec4 meta{0u};
};
static_assert(sizeof(SdfDefinitionGPU) == 48, "SdfDefinitionGPU must be 48 bytes");

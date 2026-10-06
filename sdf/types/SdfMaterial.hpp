#pragma once

// Canonical SDF material — ONE definition shared by the CPU scene builder and
// the GPU shaders (GLSL twin: shaders/types/SdfMaterial.glsl, identical
// layout). Uploaded verbatim as an array into the set=1 binding=2 SSBO
// (std430; array stride = sizeof).
//
// baseColor is one conceptual vector (rgba); emission is the rgb color plus
// an independent intensity scalar; all remaining values are independent
// shading scalars.
//
// Member alignment is pinned with alignas to the std430 vector rules
// (vec3/vec4 -> 16) so the layout never depends on glm packing.
#include <cstddef>
#include <cstdint>

#include <glm/glm.hpp>

#include "sdf/types/SdfMaterialType.hpp"

struct SdfMaterial {
    alignas(16) glm::vec4 baseColor{1.0f};            // offset  0  rgb + alpha
    float roughness = 0.5f;                           // offset 16
    float metallic = 0.0f;                            // offset 20
    float opacity = 1.0f;                             // offset 24
    SdfMaterialType mode = SdfMaterialType::Volume;   // offset 28
    alignas(16) glm::vec3 emission{1.0f, 0.5f, 0.1f}; // offset 32  emissive color
    float emissionIntensity = 2.0f;                   // offset 44  emissive strength
    float density = 1.0f;                             // offset 48  volumetric density
    float absorption = 0.5f;                          // offset 52
    float scattering = 0.5f;                          // offset 56
    float tempScale = 1.0f;                           // offset 60  temperature gradient scale
    float noiseScale = 2.5f;                          // offset 64
    float turbulence = 0.6f;                          // offset 68
    float riseSpeed = 1.5f;                           // offset 72
    // offset 76..80 is padding: the struct size must be a multiple of 16.
};
static_assert(sizeof(SdfMaterial) == 80, "SdfMaterial must be 80 bytes");
static_assert(offsetof(SdfMaterial, baseColor) == 0, "baseColor offset");
static_assert(offsetof(SdfMaterial, roughness) == 16, "roughness offset");
static_assert(offsetof(SdfMaterial, metallic) == 20, "metallic offset");
static_assert(offsetof(SdfMaterial, opacity) == 24, "opacity offset");
static_assert(offsetof(SdfMaterial, mode) == 28, "mode offset");
static_assert(offsetof(SdfMaterial, emission) == 32, "emission offset");
static_assert(offsetof(SdfMaterial, emissionIntensity) == 44, "emissionIntensity offset");
static_assert(offsetof(SdfMaterial, density) == 48, "density offset");
static_assert(offsetof(SdfMaterial, absorption) == 52, "absorption offset");
static_assert(offsetof(SdfMaterial, scattering) == 56, "scattering offset");
static_assert(offsetof(SdfMaterial, tempScale) == 60, "tempScale offset");
static_assert(offsetof(SdfMaterial, noiseScale) == 64, "noiseScale offset");
static_assert(offsetof(SdfMaterial, turbulence) == 68, "turbulence offset");
static_assert(offsetof(SdfMaterial, riseSpeed) == 72, "riseSpeed offset");

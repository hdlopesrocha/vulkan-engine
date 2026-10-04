#pragma once

// Shading / volumetric parameters for one material palette entry
// (std430, 80 bytes, vec4-aligned).
#include <glm/glm.hpp>
#include <cstdint>

struct SdfMaterialGPU {
    glm::vec4 baseColor{1.0f};
    glm::vec4 surfaceParams{0.0f}; // x = roughness, y = metallic, z = opacity, w = mode (float of SdfMaterialType)
    glm::vec4 emission{0.0f};      // rgb = color, w = intensity
    glm::vec4 volumeParams{0.0f};  // x = density, y = absorption, z = scattering, w = temperature scale
    glm::vec4 extra{0.0f};         // x = smoothK, y = noiseScale, z = turbulence, w = riseSpeed
};
static_assert(sizeof(SdfMaterialGPU) == 80, "SdfMaterialGPU must be 80 bytes");
static_assert(sizeof(SdfMaterialGPU) % 16 == 0, "SdfMaterialGPU must be multiple of 16");

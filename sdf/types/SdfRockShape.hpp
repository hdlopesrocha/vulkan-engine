#pragma once

// Rock boulder definition/material parameters. CPU-only build input: packed
// into the rock SdfDefinition (params0) and SdfMaterial (texture layer +
// tiling + pbr scalars) by createRocksFromAnchors. GLSL twin of the shape:
// includes/sdf/SdfRock.glsl (SDF_PRIM_ROCK).
#include <glm/glm.hpp>

namespace sdf_gpu {

struct RockShape {
    float noiseScale = 2.5f;      // noise frequency per local unit
    float noiseAmplitude = 0.35f; // displacement as a fraction of the radius
    float textureLayer = 7.0f;    // scene texture-array layer (-1 = flat)
    float textureTiling = 64.0f;  // world metres per texture repeat
    float roughness = 0.85f;
    float metallic = 0.0f;
    glm::vec3 tint = glm::vec3(1.0f);
};

} // namespace sdf_gpu

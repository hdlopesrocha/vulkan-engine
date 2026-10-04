#pragma once

// SDF material: what happens when a ray interacts with the SDF (separate
// from the geometry, which only determines distance).
#include "types/SdfMaterialType.hpp"
#include <glm/glm.hpp>
#include <cstdint>

namespace sdf_gpu {

struct Material {
    SdfMaterialType mode = SdfMaterialType::Volume;
    glm::vec4 baseColor = glm::vec4(1.0f);
    float roughness = 0.5f;
    float metallic = 0.0f;
    float opacity = 1.0f;
    glm::vec4 emission = glm::vec4(1.0f, 0.5f, 0.1f, 2.0f);
    float density = 1.0f;
    float absorption = 0.5f;
    float scattering = 0.5f;
    float tempScale = 1.0f;
    float noiseScale = 2.5f;
    float turbulence = 0.6f;
    float riseSpeed = 1.5f;
};

} // namespace sdf_gpu

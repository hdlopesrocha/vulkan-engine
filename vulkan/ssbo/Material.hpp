#pragma once
#include <cstddef>
#include <glm/glm.hpp>

// Packed material data (canonical shared layout; mirrors the GLSL Material in
// shaders/ssbo/Material.glsl — same name, field order and offsets). Uploaded
// verbatim as the materials palette; shaders read it through MaterialNamed.
struct Material {
    glm::vec4 materialFlags;   // .z = ambientFactor
    glm::vec4 mappingParams;   // x = mappingEnabled (0/1), y = tessLevel, z = invertHeight (0/1), w = tessHeightScale
    glm::vec4 specularParams;  // x = specularStrength, y = shininess
    glm::vec4 triplanarParams; // x = scaleU, y = scaleV, z = triplanarEnabled (0/1)
    glm::vec4 normalParams;   // x = flipNormalY (0/1), y = swapNormalXZ (0/1), z = invertWidth (0/1)
    glm::vec4 tessLevelParams; // x = minLevel, y = maxLevel, z = reflectionStrength, w = reserved
    glm::vec4 roughnessAOParams; // x = roughnessFactor, y = aoFactor, z = useAO (1.0/0.0)
};
static_assert(sizeof(Material) == 112, "Material must be 112 bytes");
static_assert(offsetof(Material, materialFlags) == 0, "materialFlags offset");
static_assert(offsetof(Material, mappingParams) == 16, "mappingParams offset");
static_assert(offsetof(Material, specularParams) == 32, "specularParams offset");
static_assert(offsetof(Material, triplanarParams) == 48, "triplanarParams offset");
static_assert(offsetof(Material, normalParams) == 64, "normalParams offset");
static_assert(offsetof(Material, tessLevelParams) == 80, "tessLevelParams offset");
static_assert(offsetof(Material, roughnessAOParams) == 96, "roughnessAOParams offset");

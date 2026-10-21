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
    glm::vec4 triplanarParams; // reserved (UV scale is per-vertex: Vertex::texCoord)
    glm::vec4 normalParams;   // x = flipNormalY (0/1), y = swapNormalXZ (0/1), z = invertWidth (0/1)
    glm::vec4 tessLevelParams; // x = minLevel, y = maxLevel, z = reflectionStrength, w = reserved
    glm::vec4 roughnessAOParams; // x = roughnessFactor, y = aoFactor, z = useAO (1.0/0.0)
    // Material kind / texture selection / blend mode.
    // x = MaterialKind, y = textureIndex (-1 = own layer), z reserved,
    // w = MaterialBlendMode.
    glm::vec4 layerParams;
    // Angular coordinate origin: xyz = sphere center, w = radius.
    glm::vec4 mappingOrigin;
};
static_assert(sizeof(Material) == 144, "Material must be 144 bytes");
static_assert(offsetof(Material, materialFlags) == 0, "materialFlags offset");
static_assert(offsetof(Material, mappingParams) == 16, "mappingParams offset");
static_assert(offsetof(Material, specularParams) == 32, "specularParams offset");
static_assert(offsetof(Material, triplanarParams) == 48, "triplanarParams offset");
static_assert(offsetof(Material, normalParams) == 64, "normalParams offset");
static_assert(offsetof(Material, tessLevelParams) == 80, "tessLevelParams offset");
static_assert(offsetof(Material, roughnessAOParams) == 96, "roughnessAOParams offset");
static_assert(offsetof(Material, layerParams) == 112, "layerParams offset");
static_assert(offsetof(Material, mappingOrigin) == 128, "mappingOrigin offset");

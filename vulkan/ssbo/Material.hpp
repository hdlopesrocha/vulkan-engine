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
    glm::vec4 normalParams;   // x = flipNormalY (0/1), y = swapNormalXZ (0/1), z = invertWidth (0/1)
    glm::vec4 tessLevelParams; // x = minLevel, y = maxLevel, z = reflectionStrength, w = reserved
    glm::vec4 roughnessAOParams; // x = roughnessFactor, y = aoFactor, z = useAO (1.0/0.0)
    // Material kind / texture selection / blend mode.
    // x = MaterialKind, y = textureIndex (-1 = own layer), z reserved,
    // w = MaterialBlendMode.
    glm::vec4 layerParams;
    // Angular coordinate origin: xyz = sphere center, w = radius.
    glm::vec4 mappingOrigin;
    // True raymarched displacement controls (independent of tessellation).
    // x = signed amplitude in world units (0 disables the new path),
    // y = frequency / UV-scale multiplier (>0), z = signed height offset
    // (normalized, 0.5 = neutral), w = strength [0..1] (0 disables).
    // Legacy tessellation displacement stays controlled by mappingParams
    // (mappingEnabled + tessHeightScale) and the global tessellation toggle.
    glm::vec4 displacementParams;
    // Secondary SDF pattern: x = type (0 texture, 1 triangles, 2 hexagons,
    // 3 Perlin, 4 Voronoi, 5..9 platonic solids),
    // y = peak displacement, z = world-unit pattern repeat, w = segment time.
    glm::vec4 patternParams;
    // Dragged segment: xyz = world-space velocity (m/s), w = intensity [0..1].
    glm::vec4 segmentParams;
};
static_assert(sizeof(Material) == 176, "Material must be 176 bytes");
static_assert(offsetof(Material, materialFlags) == 0, "materialFlags offset");
static_assert(offsetof(Material, mappingParams) == 16, "mappingParams offset");
static_assert(offsetof(Material, specularParams) == 32, "specularParams offset");
static_assert(offsetof(Material, normalParams) == 48, "normalParams offset");
static_assert(offsetof(Material, tessLevelParams) == 64, "tessLevelParams offset");
static_assert(offsetof(Material, roughnessAOParams) == 80, "roughnessAOParams offset");
static_assert(offsetof(Material, layerParams) == 96, "layerParams offset");
static_assert(offsetof(Material, mappingOrigin) == 112, "mappingOrigin offset");
static_assert(offsetof(Material, displacementParams) == 128, "displacementParams offset");
static_assert(offsetof(Material, patternParams) == 144, "patternParams offset");
static_assert(offsetof(Material, segmentParams) == 160, "segmentParams offset");

#ifndef MATERIAL_GLSL
#define MATERIAL_GLSL

// Extracted from shaders/includes/SceneBindings.glsl (single-struct GLSL type).

// Packed material data uploaded once to GPU. Matches the CPU-side Material
// (vulkan/ssbo/Material.hpp) — canonical shared layout, std430 vec4 stride.
struct Material {
    vec4 materialFlags;    // .x = skipEnvMap (set during cubemap capture), .z = ambientFactor
    vec4 mappingParams;    // x = mappingEnabled (0/1), y = tessLevel, z = invertHeight (0/1), w = tessHeightScale
    vec4 specularParams;   // x = specularStrength, y = shininess
    vec4 normalParams;     // x = flipNormalY (0/1), y = swapNormalXZ (0/1), z = invertWidth (0/1)
    vec4 tessLevelParams;  // x = minLevel, y = maxLevel, z = reflectionStrength, w = reserved
    vec4 roughnessAOParams; // x = roughnessFactor, y = aoFactor, z = useAO (1.0/0.0)
    vec4 layerParams;      // x = MaterialKind, y = textureIndex (-1 = own), z reserved, w = MaterialBlendMode
    vec4 mappingOrigin;    // xyz = Angular sphere center, w = radius
    // True raymarched displacement (independent of tessellation):
    // x = signed amplitude (world units), y = frequency / UV multiplier,
    // z = normalized height offset, w = strength [0..1].
    vec4 displacementParams;
    // Secondary SDF pattern: x = type (0 texture, 1 triangles, 2 hexagons,
    // 3 Perlin, 4 Voronoi),
    // y = peak displacement, z = world-unit repeat, w = segment time.
    vec4 patternParams;
    // Dragged segment: xyz = world-space velocity (m/s), w = intensity [0..1].
    vec4 segmentParams;
};

#endif // MATERIAL_GLSL

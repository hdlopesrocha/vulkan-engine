#ifndef MATERIAL_GLSL
#define MATERIAL_GLSL

// Extracted from shaders/includes/SceneBindings.glsl (single-struct GLSL type).

// Packed material data uploaded once to GPU. Matches the CPU-side Material
// (vulkan/ssbo/Material.hpp) — canonical shared layout, std430 vec4 stride.
struct Material {
    vec4 materialFlags;    // .x = skipEnvMap (set during cubemap capture), .z = ambientFactor
    vec4 mappingParams;    // x = mappingEnabled (0/1), y = tessLevel, z = invertHeight (0/1), w = tessHeightScale
    vec4 specularParams;   // x = specularStrength, y = shininess
    vec4 triplanarParams;  // x = scaleU, y = scaleV, z = triplanarEnabled (0/1)
    vec4 normalParams;     // x = flipNormalY (0/1), y = swapNormalXZ (0/1), z = invertWidth (0/1)
    vec4 tessLevelParams;  // x = minLevel, y = maxLevel, z = reflectionStrength, w = reserved
    vec4 roughnessAOParams; // x = roughnessFactor, y = aoFactor, z = useAO (1.0/0.0)
};

#endif // MATERIAL_GLSL

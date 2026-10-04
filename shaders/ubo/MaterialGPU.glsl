#ifndef MATERIAL_G_P_U_GLSL
#define MATERIAL_G_P_U_GLSL

// Extracted from shaders/includes/ubo.glsl (single-struct GLSL type).

// Packed material data uploaded once to GPU. Matches the CPU-side MaterialGPU (6 vec4s).
// Access this as `materials[brushIndex]` from shaders. Uses std430 for tightly-packed vec4 alignment.
struct MaterialGPU {
    vec4 materialFlags;    // .x = skipEnvMap (set during cubemap capture), .z = ambientFactor
    vec4 mappingParams;    // x = mappingEnabled (0/1), y = tessLevel, z = invertHeight (0/1), w = tessHeightScale
    vec4 specularParams;   // x = specularStrength, y = shininess
    vec4 triplanarParams;  // x = scaleU, y = scaleV, z = triplanarEnabled (0/1)
    vec4 normalParams;     // x = flipNormalY (0/1), y = swapNormalXZ (0/1), z = invertWidth (0/1)
    vec4 tessLevelParams;  // x = minLevel, y = maxLevel, z = reflectionStrength, w = reserved
    vec4 roughnessAOParams; // x = roughnessFactor, y = aoFactor, z = useAO (1.0/0.0)
};

#endif // MATERIAL_G_P_U_GLSL

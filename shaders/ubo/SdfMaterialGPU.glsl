#ifndef SDF_MATERIAL_G_P_U_GLSL
#define SDF_MATERIAL_G_P_U_GLSL

// Extracted from shaders/includes/sdf_material.glsl (single-struct GLSL type).

struct SdfMaterialGPU {
    vec4 baseColor;
    vec4 surfaceParams; // x=roughness, y=metallic, z=opacity, w=mode float
    vec4 emission;      // rgb=color, w=intensity
    vec4 volumeParams;  // x=density, y=absorption, z=scattering, w=tempScale
    vec4 extra;         // x=smoothK, y=noiseScale, z=turbulence, w=riseSpeed
};

#endif // SDF_MATERIAL_G_P_U_GLSL

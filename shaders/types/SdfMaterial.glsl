#ifndef SDF_MATERIAL_GLSL
#define SDF_MATERIAL_GLSL

// Canonical SDF material (std430, 80 B). Single definition shared with the
// CPU scene builder: sdf/types/SdfMaterial.hpp — identical layout, names
// and offsets.
// Offset table (verified against the C++ static_asserts):
//   baseColor 0, roughness 16, metallic 20, opacity 24, mode 28,
//   emission 32, emissionIntensity 44, density 48, absorption 52,
//   scattering 56, tempScale 60, noiseScale 64, turbulence 68, riseSpeed 72.
// 76..80 is std430 padding (struct size multiple of 16).
// mode ids: shaders/types/SdfMaterialType.glsl (SDF_MAT_*).
struct SdfMaterial {
    vec4 baseColor;        // offset  0  rgb + alpha
    float roughness;       // offset 16
    float metallic;        // offset 20
    float opacity;         // offset 24
    uint mode;             // offset 28
    vec3 emission;         // offset 32  emissive color
    float emissionIntensity; // offset 44
    float density;         // offset 48  volumetric density
    float absorption;      // offset 52
    float scattering;      // offset 56
    float tempScale;       // offset 60
    float noiseScale;      // offset 64
    float turbulence;      // offset 68
    float riseSpeed;       // offset 72
};

#endif // SDF_MATERIAL_GLSL

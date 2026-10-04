#ifndef SDF_INSTANCE_G_P_U_GLSL
#define SDF_INSTANCE_G_P_U_GLSL

// Extracted from shaders/includes/sdf_material.glsl (single-struct GLSL type).

struct SdfInstanceGPU {
    vec4 posScale;   // xyz=world pos, w=uniform scale
    vec4 rotSeed;    // xyz=euler XYZ radians, w=animation seed
    vec4 sizeParams; // x=heightScale, y=radiusScale, z=intensity, w=unused
    uvec4 indices;   // x=defIdx, y=matIdx, z=containerIdx, w=flags
    vec4 boundsMin;  // world AABB min
    vec4 boundsMax;  // world AABB max
};

#endif // SDF_INSTANCE_G_P_U_GLSL

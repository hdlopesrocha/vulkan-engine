#ifndef SDF_CONTAINER_G_P_U_GLSL
#define SDF_CONTAINER_G_P_U_GLSL

// Extracted from shaders/includes/sdf_material.glsl (single-struct GLSL type).

struct SdfContainerGPU {
    vec4 boundsMin;
    vec4 boundsMax;
    uvec4 gridInfo;   // x,y,z=resolution, w=global index-buffer start
    uvec4 gridOffset; // x=global cell-buffer start, y=index start, z=count, w=flags
};

#endif // SDF_CONTAINER_G_P_U_GLSL

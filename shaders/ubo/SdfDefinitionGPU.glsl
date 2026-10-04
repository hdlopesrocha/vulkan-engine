#ifndef SDF_DEFINITION_G_P_U_GLSL
#define SDF_DEFINITION_G_P_U_GLSL

// Extracted from shaders/includes/sdf_material.glsl (single-struct GLSL type).

struct SdfDefinitionGPU {
    vec4 params0;
    vec4 params1;
    uvec4 meta; // x=prim(SDF_PRIM_*), y=op(SDF_OP_*), z=deformFlags, w=bitcast float smoothK
};

#endif // SDF_DEFINITION_G_P_U_GLSL

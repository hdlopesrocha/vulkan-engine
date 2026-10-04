#ifndef SDF_PARAMS_U_B_O_GLSL
#define SDF_PARAMS_U_B_O_GLSL

// Extracted from shaders/includes/sdf_material.glsl (single-struct GLSL type).

struct SdfParamsUBO {
    vec4 timeDebug;   // x=time, y=packedModeDebug(float bits of (mode<<16|flags)), z=maxSteps, w=safety
    vec4 marchParams; // x=minStep, y=maxStep, z=epsilon, w=earlyTerm opacity threshold
    vec4 fireColors0; // reserved gradient low (generic)
    vec4 fireColors1; // reserved gradient high (generic)
};

#endif // SDF_PARAMS_U_B_O_GLSL

#ifndef DEBUG_SDF_INSTANCE_DATA_GLSL
#define DEBUG_SDF_INSTANCE_DATA_GLSL

// Extracted from shaders/DebugSDFRenderer.vert (single-struct GLSL type).

struct DebugSdfInstanceData {
    mat4 model;
    vec4 sdf0;
    vec4 sdf1;
    vec4 meta; // meta.x = brushIndex (stored as float)
};

#endif // DEBUG_SDF_INSTANCE_DATA_GLSL

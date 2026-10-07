#ifndef DEBUG_CUBE_INSTANCE_DATA_GLSL
#define DEBUG_CUBE_INSTANCE_DATA_GLSL

// Extracted from shaders/DebugCubeRenderer.vert (single-struct GLSL type).

struct DebugCubeInstanceData {
    mat4 model;
    vec4 color;  // vec4 for proper alignment
};

#endif // DEBUG_CUBE_INSTANCE_DATA_GLSL

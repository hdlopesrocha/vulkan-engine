#ifndef DEBUG_CUBE_INSTANCE_DATA_NAMED_GLSL
#define DEBUG_CUBE_INSTANCE_DATA_NAMED_GLSL

// Extracted from shaders/DebugCubeRenderer.vert (single-struct GLSL type).

// Named view over the packed InstanceData - same data, descriptive names. The builder below is the
// only place the packed component letters are read; every other access uses the
// named attributes.
struct DebugCubeInstanceDataNamed {
    mat4 model;
    vec3 color;
};

#endif // DEBUG_CUBE_INSTANCE_DATA_NAMED_GLSL

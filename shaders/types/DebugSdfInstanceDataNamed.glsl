#ifndef DEBUG_SDF_INSTANCE_DATA_NAMED_GLSL
#define DEBUG_SDF_INSTANCE_DATA_NAMED_GLSL

// Extracted from shaders/DebugSDFRenderer.vert (single-struct GLSL type).

// Named view over the packed InstanceData - same data, descriptive names. The builder below is the
// only place the packed component letters are read; every other access uses the
// named attributes.
// sdf0/sdf1 pack the 8 box-corner SDF values, meta.x carries the brush index.
struct DebugSdfInstanceDataNamed {
    mat4 model;
    float sdfCorner0;
    float sdfCorner1;
    float sdfCorner2;
    float sdfCorner3;
    float sdfCorner4;
    float sdfCorner5;
    float sdfCorner6;
    float sdfCorner7;
    int brushIndex;
};

#endif // DEBUG_SDF_INSTANCE_DATA_NAMED_GLSL

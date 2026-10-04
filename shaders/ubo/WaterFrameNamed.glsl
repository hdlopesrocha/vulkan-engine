#ifndef WATER_FRAME_NAMED_GLSL
#define WATER_FRAME_NAMED_GLSL

// Extracted from shaders/postprocess.frag (single-struct GLSL type).

// Named view over the packed WaterUBO - same data, descriptive names. The builder below is the
// only place the packed component letters are read; every other access uses the
// named attributes.
struct WaterFrameNamed {
    mat4 viewProjection;
    mat4 invViewProjection;
    vec3 viewPosition;
    vec2 screenSize;
    vec2 invScreenSize;
    float brushAlpha;
    float brushMode;
    float waterBlurEnabled;
    // M12 (perf report 22): vegetation offscreen targets are downscaled, so
    // the composite must take the closest of the 2x2 depth taps (see the
    // packed member's comment in postprocess.frag). Was the std140 pad.
    float vegetationScaled;
};

#endif // WATER_FRAME_NAMED_GLSL

#ifndef WATER_RENDER_PARAMS_NAMED_GLSL
#define WATER_RENDER_PARAMS_NAMED_GLSL

// Extracted from shaders/includes/ubo.glsl (single-struct GLSL type).

// Named view over the packed WaterRenderUBO - same data, descriptive names. The builder below is the
// only place the packed component letters are read; every other access uses the
// named attributes.
struct WaterRenderParamsNamed {
    float waterTime;
    bool refractionAllowed;
    bool reflectionAllowed;
    bool blurAllowed;
    bool solidDepthIsCurrent;
};

#endif // WATER_RENDER_PARAMS_NAMED_GLSL

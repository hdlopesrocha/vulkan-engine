#ifndef WIND_RAYMARCH_U_B_O_GLSL
#define WIND_RAYMARCH_U_B_O_GLSL

// Exact GLSL mirror of vulkan/ubo/WindRaymarchUBO.hpp (std140, 112 B).
// Offsets: invViewProjection 0, surface 64, viewport 80, cameraPosition 96.
struct WindRaymarchUBO {
    mat4 invViewProjection; // offset  0  clip -> world
    vec4 surface;           // offset 64  x = target speed (m/s), y = max distance (m),
                            //            z = steps, w = opacity
    vec4 viewport;          // offset 80  x = width (px), y = height (px),
                            //            z = wind clock (s), w = unused
    vec4 cameraPosition;    // offset 96  xyz = camera world position, w unused
};

#endif // WIND_RAYMARCH_U_B_O_GLSL

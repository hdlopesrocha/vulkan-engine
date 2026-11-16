#ifndef WIND_PARTICLES_U_B_O_GLSL
#define WIND_PARTICLES_U_B_O_GLSL

// Exact GLSL mirror of vulkan/ubo/WindParticlesUBO.hpp (std140 uniform block
// member). Offsets must match the C++ side 1:1:
//   viewProjection 0, invViewProjection 64, timing 128, trail 144, color 160,
//   spawn 176, cameraPosition 192, extra 208, total size 224 bytes.
// EVERY component is documented here; keep in sync with the .hpp.
struct WindParticlesUBO {
    mat4 viewProjection;    // offset   0  world -> clip (frustum test)
    mat4 invViewProjection; // offset  64  clip -> world (respawn inside)
    vec4 timing;            // offset 128  x = wind clock (s), y = dt (s),
                            //             z = particle count, w = spawn seed
    vec4 trail;             // offset 144  x = duration (s), y = sample interval (s),
                            //             z = ring capacity, w = width (px)
    vec4 style;             // offset 160  x = speed for full red (m/s),
                            //             y = peak opacity, zw reserved
    vec4 spawn;             // offset 176  x = spawn near (m), y = spawn far plane (m),
                            //             z = viewport width (px), w = height (px)
    vec4 cameraPosition;    // offset 192  xyz = camera world position (respawn ray
                            //             origin + far-plane test), w unused
    vec4 extra;             // offset 208  x = frustum margin, y = fade-in (s),
                            //             z = wind enabled flag, w reserved
};

#endif // WIND_PARTICLES_U_B_O_GLSL

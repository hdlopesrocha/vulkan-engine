#ifndef WIND_PARTICLE_GLSL
#define WIND_PARTICLE_GLSL

// Per-particle state + trail ring layout shared by the wind simulation
// (shaders/computation/wind/WindParticles.comp) and the trail renderer
// (shaders/renderer/wind/WindTrail.vert). std430, 48 bytes per particle.
//
// Keep WIND_TRAIL_CAPACITY in sync with:
//   WindRenderer::kTrailCapacity (vulkan/renderer/wind/WindRenderer.hpp)
//   shaders/ubo/WindParticlesUBO.glsl  (trail.z repeats it at dispatch time)
#define WIND_TRAIL_CAPACITY 192

struct WindParticle {
    // xyz = world position, w = seconds since the last (re)spawn.
    vec4 position;
    // x = trail ring head (newest sample), y = valid sample count (<= capacity),
    // z = per-particle RNG seed, w = flags (bit 0 = initialised/spawned).
    uvec4 trail;
    // x = clock time of the last appended sample (s); yzw reserved.
    vec4 sampleTime;
};

#endif // WIND_PARTICLE_GLSL

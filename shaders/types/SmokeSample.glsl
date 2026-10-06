#ifndef SMOKE_SAMPLE_GLSL
#define SMOKE_SAMPLE_GLSL

// Shader-internal value type (no CPU mirror; sdf_smoke.glsl output).

// Full per-sample smoke evaluation (march path): base density at the
// bullet-displaced position, tunnel/wake thinning, wave modulation,
// temporal dissipation. Debug metrics are returned for the §24 views.
struct SmokeSample {
    float density;  // base layered density (pre-bullet, debug view 2)
    float sdf;      // grown sphere SDF (for debug)
    float bullet;   // min bullet-path SDF (for debug)
    float tunnel;
    float pressure;
    float wave;
    float turb;
    float wake;
    float finalD;   // post-tunnel/wave/fade density (debug view 9)
    float heat;     // hot-air mask 0..1 (tunnel core): thins smoke, glows
};

#endif // SMOKE_SAMPLE_GLSL

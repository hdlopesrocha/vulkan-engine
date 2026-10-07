#ifndef WATER_VERTEX_WAVE_GLSL
#define WATER_VERTEX_WAVE_GLSL

// Extracted from shaders/includes/water/WaterTese.glsl (single-struct GLSL type).

// ── Shared per-vertex wave core ─────────────────────────────────────────
// Displaces `pos` along its base normal by the thickness-zoned wave field and
// returns the analytic wave normal (exact for the `base + N * h` height
// field). The TES full path below and the WATER_NO_TESS vertex shader both
// call this, so the two geometry paths can never drift.
struct WaterVertexWave {
    vec3 pos;           // displaced world position
    vec3 normal;        // analytic wave normal at the displaced surface
    vec3 basePos;       // undisplaced base world position
    float displacement; // signed height displacement along the base normal
    float worldHeight;  // pos.y: world-space height of the displaced surface
};

#endif // WATER_VERTEX_WAVE_GLSL

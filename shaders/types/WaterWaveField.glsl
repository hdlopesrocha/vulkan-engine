#ifndef WATER_WAVE_FIELD_GLSL
#define WATER_WAVE_FIELD_GLSL

// Extracted from shaders/includes/water/Gerstner.glsl (single-struct GLSL type).

struct WaterWaveField {
    float height;     // vertical displacement along the base normal
    vec3  grad;       // analytic d(height)/d(world position), y = 0 (height field)
    float foam;       // 0..1 whitewater coverage (breaking driven)
    float contact;    // 0..1 shoreline contact foam (water meets solid at depth 0)
    float calmMask;   // 1 when the field ran (waves active), 0 when it early-outed
    float swell;      // signed sine profile (the shore swell)
    float swellSlope; // its along-shore slope (the cosine)
    float breaking;   // unused (one sine, no breaking) - kept for the debug views
    float steepness;  // unused (one sine, no steepening) - kept for the debug views
    vec2  disp;       // Gerstner horizontal displacement (world xz) - applied by the vertex stages
};

#endif // WATER_WAVE_FIELD_GLSL

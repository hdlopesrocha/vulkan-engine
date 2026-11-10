#ifndef SDF_EFFECTS_COMMON_GLSL
#define SDF_EFFECTS_COMMON_GLSL

// Shared helpers for the CPU brush effect mirrors. Keep these formulas in
// sync with sdf/effects/*DistanceEffect.cpp.
float sdfEffectBc(float c, float brightness, float contrast) {
    c = clamp(c + brightness, -1.0, 1.0);
    return clamp(c * contrast, -1.0, 1.0);
}

#endif // SDF_EFFECTS_COMMON_GLSL

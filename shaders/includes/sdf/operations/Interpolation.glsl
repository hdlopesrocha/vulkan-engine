#ifndef SDF_OPERATION_INTERPOLATION_GLSL
#define SDF_OPERATION_INTERPOLATION_GLSL

// Linear blend between two fields: t=0 keeps d1, t=1 keeps d2. The factor is
// clamped to [0,1] so the result stays 1-Lipschitz when both operands are
// signed distance fields (the generic call site allows k up to 2).
float opInterpolation(float d1, float d2, float k) {
    return mix(d1, d2, clamp(k, 0.0, 1.0));
}

#endif // SDF_OPERATION_INTERPOLATION_GLSL

#ifndef SDF_OPERATION_SMOOTH_SUBTRACTION_GLSL
#define SDF_OPERATION_SMOOTH_SUBTRACTION_GLSL

// Polynomial smooth subtraction (IQ). k is clamped to [0, 2]; k <= 0 falls
// back to the hard subtraction.
float opSmoothSubtraction(float d1, float d2, float k) {
    float kk = clamp(k, 0.0, 2.0);
    if (kk <= 1e-6) {
        return max(d2, -d1);
    }
    float h = clamp(0.5 - 0.5 * (d2 + d1) / kk, 0.0, 1.0);
    return mix(d2, -d1, h) + kk * h * (1.0 - h);
}

#endif // SDF_OPERATION_SMOOTH_SUBTRACTION_GLSL

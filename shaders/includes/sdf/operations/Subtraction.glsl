#ifndef SDF_OPERATION_SUBTRACTION_GLSL
#define SDF_OPERATION_SUBTRACTION_GLSL

// Subtracts d1 from d2 (keeps the part of d2 outside d1).
float opSubtraction(float d1, float d2) {
    return max(d2, -d1);
}

#endif // SDF_OPERATION_SUBTRACTION_GLSL

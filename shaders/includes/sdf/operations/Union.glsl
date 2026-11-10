#ifndef SDF_OPERATION_UNION_GLSL
#define SDF_OPERATION_UNION_GLSL

float opUnion(float d1, float d2) {
    return min(d1, d2);
}

#endif // SDF_OPERATION_UNION_GLSL

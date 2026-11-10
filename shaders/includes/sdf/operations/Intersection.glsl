#ifndef SDF_OPERATION_INTERSECTION_GLSL
#define SDF_OPERATION_INTERSECTION_GLSL

float opIntersection(float d1, float d2) {
    return max(d1, d2);
}

#endif // SDF_OPERATION_INTERSECTION_GLSL

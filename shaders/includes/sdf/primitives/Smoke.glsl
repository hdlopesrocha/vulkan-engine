#ifndef SDF_PRIMITIVE_SMOKE_GLSL
#define SDF_PRIMITIVE_SMOKE_GLSL

#include "Sphere.glsl"

// Smoke's static field is the maximum-radius sphere; the growth over time is
// applied in smokeMarchSDF (sdfEvalInstance branches before the dispatcher).
float sdSmoke(vec3 p, vec4 p0, vec4 p1) {
    return sdSphere(p, p0, p1);
}

#endif // SDF_PRIMITIVE_SMOKE_GLSL

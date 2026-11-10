#ifndef SDF_PRIMITIVE_SPHERE_GLSL
#define SDF_PRIMITIVE_SPHERE_GLSL

// Sphere. p0.x = radius.
float sdSphere(vec3 p, vec4 p0, vec4 p1) {
    return length(p) - max(p0.x, 0.0);
}

#endif // SDF_PRIMITIVE_SPHERE_GLSL

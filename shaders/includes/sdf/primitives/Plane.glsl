#ifndef SDF_PRIMITIVE_PLANE_GLSL
#define SDF_PRIMITIVE_PLANE_GLSL

// Plane. p0.xyz = normal (normalized internally), p0.w = offset.
// Signed distance = dot(p, n) - offset.
float sdPlane(vec3 p, vec4 p0, vec4 p1) {
    float l = length(p0.xyz);
    vec3 n = (l > 1e-8) ? (p0.xyz / l) : vec3(0.0, 1.0, 0.0);
    return dot(p, n) - p0.w;
}

#endif // SDF_PRIMITIVE_PLANE_GLSL

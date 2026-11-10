#ifndef SDF_PRIMITIVE_TORUS_GLSL
#define SDF_PRIMITIVE_TORUS_GLSL

// Torus around Y. p0.xy = vec2(R, r) (major, minor radii).
float sdTorus(vec3 p, vec4 p0, vec4 p1) {
    float R = max(p0.x, 0.0);
    float r = max(p0.y, 0.0);
    vec2 q = vec2(length(p.xz) - R, p.y);
    return length(q) - r;
}

#endif // SDF_PRIMITIVE_TORUS_GLSL

#ifndef SDF_PRIMITIVE_CAPSULE_GLSL
#define SDF_PRIMITIVE_CAPSULE_GLSL

// Y-aligned capsule. p0.x = radius, p0.y = half height of the cylindrical
// section (total height = 2*halfHeight + 2*radius).
float sdCapsule(vec3 p, vec4 p0, vec4 p1) {
    float r = max(p0.x, 0.0);
    float hh = max(p0.y, 0.0);
    vec3 pa = vec3(p.x, p.y - hh, p.z);
    vec3 ba = vec3(0.0, 2.0 * hh, 0.0);
    float denom = dot(ba, ba);
    if (denom < 1e-12) {
        return length(p) - r; // degenerate: sphere
    }
    float h = clamp(dot(pa, ba) / denom, 0.0, 1.0);
    return length(pa - ba * h) - r;
}

#endif // SDF_PRIMITIVE_CAPSULE_GLSL

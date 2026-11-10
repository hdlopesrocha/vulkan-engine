#ifndef SDF_PRIMITIVE_CYLINDER_GLSL
#define SDF_PRIMITIVE_CYLINDER_GLSL

// Y-aligned cylinder. p0.x = radius, p0.y = half height (centered).
float sdCylinder(vec3 p, vec4 p0, vec4 p1) {
    float r = max(p0.x, 0.0);
    float h = max(p0.y, 1e-6);
    vec2 d = vec2(length(p.xz) - r, abs(p.y) - h * 0.5);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

#endif // SDF_PRIMITIVE_CYLINDER_GLSL

#ifndef SDF_PRIMITIVE_ROUNDED_BOX_GLSL
#define SDF_PRIMITIVE_ROUNDED_BOX_GLSL

// Box with rounded corners. p0.xyz = outer half extents, p1.x = corner radius.
float sdRoundedBox(vec3 p, vec4 p0, vec4 p1) {
    vec3 b = max(p0.xyz, vec3(0.0));
    float r = clamp(p1.x, 0.0, min(b.x, min(b.y, b.z)));
    vec3 q = abs(p) - max(b - vec3(r), vec3(0.0));
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

#endif // SDF_PRIMITIVE_ROUNDED_BOX_GLSL

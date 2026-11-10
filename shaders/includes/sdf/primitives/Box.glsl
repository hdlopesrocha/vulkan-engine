#ifndef SDF_PRIMITIVE_BOX_GLSL
#define SDF_PRIMITIVE_BOX_GLSL

// Axis-aligned box. p0.xyz = half extents.
float sdBox(vec3 p, vec4 p0, vec4 p1) {
    vec3 b = max(p0.xyz, vec3(0.0));
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

#endif // SDF_PRIMITIVE_BOX_GLSL

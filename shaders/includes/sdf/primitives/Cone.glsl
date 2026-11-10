#ifndef SDF_PRIMITIVE_CONE_GLSL
#define SDF_PRIMITIVE_CONE_GLSL

// Round cone (IQ, exact): base disk radius r1 at y=0, tip radius r2 at y=h.
// Our cone: base radius r at y=-h/2, apex (r2=0) at y=+h/2. For squat cones
// (r >= h) the round-cone slope factor breaks down, so fall back to the
// bounding capped cylinder: it contains the cone, hence its SDF is a safe
// lower bound for sphere tracing (never overestimates the true distance).
float sdCone(vec3 p, vec4 p0, vec4 p1) {
    float r = max(p0.x, 0.0);
    float h = max(p0.y, 1e-6);
    float b = r / h;
    if (b < 0.999) {
        vec2 q = vec2(length(p.xz), p.y + h * 0.5);
        float a = sqrt(max(1.0 - b * b, 1e-9));
        float k = dot(q, vec2(-b, a));
        if (k < 0.0) {
            return length(q) - r;
        }
        if (k > a * h) {
            return length(q - vec2(0.0, h));
        }
        return dot(q, vec2(a, b)) - r;
    }
    vec2 d = vec2(length(p.xz) - r, abs(p.y) - h * 0.5);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

#endif // SDF_PRIMITIVE_CONE_GLSL

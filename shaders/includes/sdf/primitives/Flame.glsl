#ifndef SDF_PRIMITIVE_FLAME_GLSL
#define SDF_PRIMITIVE_FLAME_GLSL

// Tapered base-anchored flame: exact round cone (IQ) with sphere caps from
// the base disk (y=0, radius r1=p0.x) to the tip (y=h=p0.y, radius r2=p1.x).
// A rounded capsule with two different radii: r2 == r1 is a capsule,
// r2 == 0 a sharp cone tip. Flames start at the instance point (+Y axis).
float sdTaperedFlame(vec3 p, vec4 p0, vec4 p1) {
    float r1 = max(p0.x, 0.0);
    float r2 = max(p1.x, 0.0);
    float h = max(p0.y, 1e-6);
    float b = (r1 - r2) / h;
    float a = sqrt(max(1.0 - b * b, 1e-9));
    vec2 q = vec2(length(p.xz), p.y);
    float k = dot(q, vec2(-b, a));
    if (k < 0.0) {
        return length(q) - r1; // below the base disk
    }
    if (k > a * h) {
        return length(q - vec2(0.0, h)); // above the tip
    }
    return dot(q, vec2(a, b)) - r1;
}

#endif // SDF_PRIMITIVE_FLAME_GLSL

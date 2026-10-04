// Generic SDF primitive library (exact IQ-style formulas).
// All primitives share one signature so the dispatcher can evaluate any
// definition: float sdfX(vec3 p, vec4 p0, vec4 p1).
//
// Parameter packing (matches types/Sdf*GPU.hpp + sdf/gpu/SdfScene.cpp):
//   sphere:      p0.x = radius
//   box:         p0.xyz = half extents
//   rounded box: p0.xyz = half extents (outer), p1.x = corner radius
//   capsule:     Y-aligned: p0.x = radius, p0.y = half height of the
//                cylindrical section (total height = 2*halfHeight + 2*radius)
//   cylinder:    p0.x = radius, p0.y = half height (Y axis, centered)
//   cone:        p0.x = base radius, p0.y = full height (base at -h/2, apex +h/2)
//   torus:       p0.xy = vec2(R, r) (major, minor radii, around Y)
//   plane:       p0.xyz = normal (normalized internally), p0.w = offset
//                (signed distance = dot(p, n) - offset)
// Unused components are ignored. No textures, no vendor-specific behavior.

#ifndef SDF_PRIMITIVES_GLSL
#define SDF_PRIMITIVES_GLSL

#define SDF_PRIM_SPHERE 0u
#define SDF_PRIM_BOX 1u
#define SDF_PRIM_ROUNDED_BOX 2u
#define SDF_PRIM_CAPSULE 3u
#define SDF_PRIM_CYLINDER 4u
#define SDF_PRIM_CONE 5u
#define SDF_PRIM_TORUS 6u
#define SDF_PRIM_PLANE 7u
#define SDF_PRIM_FLAME 8u
#define SDF_PRIM_SMOKE 9u

float sdSphere(vec3 p, vec4 p0, vec4 p1) {
    return length(p) - max(p0.x, 0.0);
}

float sdBox(vec3 p, vec4 p0, vec4 p1) {
    vec3 b = max(p0.xyz, vec3(0.0));
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdRoundedBox(vec3 p, vec4 p0, vec4 p1) {
    vec3 b = max(p0.xyz, vec3(0.0));
    float r = clamp(p1.x, 0.0, min(b.x, min(b.y, b.z)));
    vec3 q = abs(p) - max(b - vec3(r), vec3(0.0));
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

float sdCapsule(vec3 p, vec4 p0, vec4 p1) {
    // Y-aligned capsule matching the CPU SdfScene packing (radius, halfHeight).
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

float sdCylinder(vec3 p, vec4 p0, vec4 p1) {
    float r = max(p0.x, 0.0);
    float h = max(p0.y, 1e-6);
    vec2 d = vec2(length(p.xz) - r, abs(p.y) - h * 0.5);
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}

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

float sdTorus(vec3 p, vec4 p0, vec4 p1) {
    float R = max(p0.x, 0.0);
    float r = max(p0.y, 0.0);
    vec2 q = vec2(length(p.xz) - R, p.y);
    return length(q) - r;
}

float sdPlane(vec3 p, vec4 p0, vec4 p1) {
    float l = length(p0.xyz);
    vec3 n = (l > 1e-8) ? (p0.xyz / l) : vec3(0.0, 1.0, 0.0);
    return dot(p, n) - p0.w;
}

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

// Dispatcher: evaluates the primitive selected by primType. Unknown types
// return a large distance (treated as empty space, never as a surface).
float sdfPrimitive(vec3 p, uint primType, vec4 p0, vec4 p1) {
    if (primType == SDF_PRIM_SPHERE) {
        return sdSphere(p, p0, p1);
    }
    if (primType == SDF_PRIM_BOX) {
        return sdBox(p, p0, p1);
    }
    if (primType == SDF_PRIM_ROUNDED_BOX) {
        return sdRoundedBox(p, p0, p1);
    }
    if (primType == SDF_PRIM_CAPSULE) {
        return sdCapsule(p, p0, p1);
    }
    if (primType == SDF_PRIM_CYLINDER) {
        return sdCylinder(p, p0, p1);
    }
    if (primType == SDF_PRIM_CONE) {
        return sdCone(p, p0, p1);
    }
    if (primType == SDF_PRIM_TORUS) {
        return sdTorus(p, p0, p1);
    }
    if (primType == SDF_PRIM_PLANE) {
        return sdPlane(p, p0, p1);
    }
    if (primType == SDF_PRIM_FLAME) {
        return sdTaperedFlame(p, p0, p1);
    }
    if (primType == SDF_PRIM_SMOKE) {
        // Static maximum-radius sphere; growth is applied in
        // smokeMarchSDF (sdfEvalInstance branches before dispatch).
        return sdSphere(p, p0, p1);
    }
    return 1e5;
}

#endif // SDF_PRIMITIVES_GLSL

// Shared wind-field sampling contract (set=0, binding 27).
//
// Every consumer (vegetation vertex, fire, SDF) includes this file to sample
// the IDENTICAL field: ambient base flow + Perlin gusts + up to 4 Rankine
// tornadoes, all GPU-side functions of the consumer's own clock (Bullet
// pattern: zero per-frame CPU beyond the packed UBO).
//
// Requires perlin.glsl (perlinNoise3D) to be included BEFORE this file.
// The uniform block is declared here so consumers only need this one include.
//
// Consumer API:
//   vec3 windAmbient(vec3 p, float time)              — base flow + gusts
//   vec3 tornadoVelocity(WindTornado t, vec3 p, float time) — one funnel
//   vec3 windSVF(vec3 p, float time)                   — ambient + all active
// UBO field semantics: see shaders/ubo/WindField.glsl (component letters)
//   and vulkan/ubo/WindField.hpp. The block mirror is exact (std140).
#ifndef WIND_FIELD_GLSL
#define WIND_FIELD_GLSL

#include "../ubo/WindField.glsl"

layout(set = 0, binding = 27) uniform WindFieldBlock {
    WindField windField;
};

// Ambient base flow + Perlin gusts. NaN-safe: direction re-normalized with a
// degenerate fallback, all divisors floored, inputs clamped (huge world/time
// coordinates lose float precision and poison the hash otherwise).
vec3 windAmbient(vec3 p, float time) {
    vec2 d = windField.ambientA.xy;
    float dl = length(d);
    d = (dl > 1e-4) ? d / dl : vec2(1.0, 0.0);
    float strength = max(windField.ambientA.z, 0.0);
    float speed = max(windField.ambientA.w, 0.0);
    float t = clamp(time, 0.0, 86400.0);
    vec3 pc = vec3(clamp(p.x, -1e5, 1e5), clamp(p.y, -1e4, 1e4), clamp(p.z, -1e5, 1e5));
    float bf = max(windField.ambientB.x, 1e-5);
    float gf = max(windField.ambientB.y, 1e-5);
    float gs = clamp(windField.ambientB.z, 0.0, 4.0);
    // Large-scale advected gust + finer cross detail, both time-scrolled.
    float n1 = perlinNoise3D(vec3(pc.x * bf, pc.z * bf, t * speed * 0.05));
    float n2 = perlinNoise3D(vec3(pc.x * gf + 37.7, pc.z * gf - 11.3, t * speed * 0.09 + 5.2));
    vec3 base = vec3(d.x, 0.0, d.y) * strength;
    vec2 perp = vec2(-d.y, d.x);
    vec3 gust = (vec3(d.x, 0.0, d.y) * n1 + vec3(perp.x, 0.0, perp.y) * (n2 * 0.6)) * (gs * strength);
    return base + gust + vec3(0.0, n1 * gs * strength * 0.15, 0.0);
}

// Funnel centre, fully GPU-side: base + drift * scaled time + wander orbit.
// Inactive tornadoes (c.y < 0.5) never reach here — callers skip them.
vec2 windTornadoCenter(WindTornado t, float time) {
    float tc = clamp(time, 0.0, 86400.0);
    float st = tc * clamp(t.c.x, 0.0, 8.0); // scaled time (delta_time)
    float lt = st + t.b.w;                  // + phase staggers cycles
    float wR = max(t.c.z, 0.0);
    return t.a.xy + t.d.xy * st + wR * vec2(cos(t.c.w * lt), sin(t.c.w * lt));
}

// Rankine-combined vortex: solid-body rotation inside the core (v ~ r),
// 1/r decay outside, radial inflow, and a height-profiled updraft. Direction
// sign swings dynamically unless amplitude is 0 (static direction).
// NaN-safe: radial normalize guarded, core radius floored, height floored,
// exp argument is a non-positive square (never overflows).
vec3 tornadoVelocity(WindTornado t, vec3 p, float time) {
    if (t.c.y < 0.5) return vec3(0.0);
    float R = max(t.a.w, 0.5);
    float H = max(t.b.x, 1.0);
    float S = max(t.b.y, 0.0);
    if (S <= 0.0) return vec3(0.0);
    float dirSign = clamp(t.b.z, -1.0, 1.0);
    float swingAmp = clamp(t.d.z, 0.0, 1.0);
    float tc = clamp(time, 0.0, 86400.0);
    float lt = tc * clamp(t.c.x, 0.0, 8.0) + t.b.w;
    // Static direction when amplitude is 0 (or frequency is 0); otherwise
    // blend the base sign toward a cos oscillation. dirSign == 0 keeps a
    // pure updraft column (no tangential component).
    float effSign = (swingAmp <= 0.0 || t.d.w == 0.0)
        ? ((dirSign >= 0.0) ? 1.0 : -1.0) * min(abs(dirSign), 1.0)
        : dirSign * mix(1.0, cos(t.d.w * lt), swingAmp);
    vec2 rel = clamp(p.xz, vec2(-1e5), vec2(1e5)) - windTornadoCenter(t, tc);
    float r = length(rel);
    float rSafe = max(r, 1e-3);
    vec2 radial = rel / rSafe;
    vec2 tangent = effSign * vec2(-radial.y, radial.x);
    // Rankine profile: solid body (r/R) inside the core, R/r outside.
    float tangSpeed = S * ((r < R) ? (r / R) : (R / rSafe));
    float inflow = 0.35 * tangSpeed;
    // Funnel spans groundY..groundY+height; full strength at/below the base,
    // zero at the top. Horizontal swirl persists weakly aloft.
    float hProf = clamp(1.0 - (clamp(p.y, -1e4, 1e4) - t.a.z) / H, 0.0, 1.0);
    float q = rSafe / R;
    float up = S * 0.8 * hProf * exp(-q * q);
    vec2 horiz = (tangent * tangSpeed - radial * inflow) * (0.25 + 0.75 * hProf);
    return vec3(horiz.x, up, horiz.y);
}

// Full shared field: ambient + sum of active tornadoes. The counts.x early-out
// keeps the loop cheap when fewer than 4 tornadoes are active; the per-funnel
// active-flag guard inside tornadoVelocity covers sparse activation.
vec3 windSVF(vec3 p, float time) {
    vec3 v = windAmbient(p, time);
    float n = clamp(windField.counts.x, 0.0, float(WIND_FIELD_MAX_TORNADOES));
    for (int i = 0; i < WIND_FIELD_MAX_TORNADOES; ++i) {
        if (float(i) >= n) break;
        v += tornadoVelocity(windField.tornadoes[i], p, time);
    }
    return v;
}

#endif // WIND_FIELD_GLSL

// Shared wind-field sampling contract (set=0, binding 27).
//
// Every consumer (vegetation vertex, fire, SDF, volumetric clouds) includes
// this file to sample the IDENTICAL field: a VECTOR-VALUED fBm velocity field
// (all three axes; there is no authored direction or mean flow — the local
// direction IS the fBm direction) plus up to 4 Rankine tornadoes, all GPU-side
// functions of the consumer's own clock (Bullet pattern: zero per-frame CPU
// beyond the packed UBO).
//
// Requires perlin.glsl (perlinNoise3D) to be included BEFORE this file.
// The uniform block is declared here so consumers only need this one include.
//
// Consumer API:
//   vec3 windFbm3(vec3 p, float time)                  — raw vector fBm, |v| <= 1
//   vec3 windAmbient(vec3 p, float time)               — fBm velocity (m/s)
//   vec3 tornadoVelocity(WindTornado t, vec3 p, float time) — one funnel
//   vec3 windSVF(vec3 p, float time)                   — ambient + all active
// UBO field semantics: see shaders/ubo/WindField.glsl (component letters)
//   and vulkan/ubo/WindField.hpp. The block mirror is exact (std140).
#ifndef WIND_FIELD_GLSL
#define WIND_FIELD_GLSL

#include "../../ubo/WindField.glsl"

layout(set = 0, binding = 27) uniform WindFieldBlock {
    WindField windField;
};

// Vector-valued fBm gust field: the wind velocity itself.
//
// Each component is its own octave stack sampled at a decorrelated domain
// offset, so the wind velocity varies along X, Y and Z (no axis is
// privileged) and its local direction comes straight out of the noise field.
// Parameters live in ambientB/ambientC and are edited in the Wind widget's
// "FBM Field" section:
//   ambientB.x frequency   spatial scale of the first octave (1/m)
//   ambientB.y octaves     active octave count (1..WIND_FBM_MAX_OCTAVES)
//   ambientB.z lacunarity  frequency multiplier per octave
//   ambientB.w gain        amplitude multiplier per octave (persistence)
//   ambientC.z seed        world-space domain offset
// The result is normalized by the amplitude sum, so |fbm| stays ~<= 1 for any
// octave count and the m/s amplitude in ambientC.x keeps its meaning.
//
// There is no mean flow to advect the domain, so the pattern instead morphs
// along a fixed diagonal at a rate scaled by the shared wind speed slider
// (ambientA.w).
//
// NaN-safe: inputs are clamped by the caller, frequencies are floored and the
// octave count is clamped to the compile-time loop bound.
vec3 windFbm3(vec3 p, float time) {
    // Decorrelating offsets for the Y and Z components: three far-apart
    // samples of the same Perlin lattice.
    const vec3 kComponentOffsetY = vec3(31.416, 17.234, 11.719);
    const vec3 kComponentOffsetZ = vec3(-13.137, 41.309, 7.913);

    int octaves = int(clamp(windField.ambientB.y, 1.0, float(WIND_FBM_MAX_OCTAVES)) + 0.5);
    float frequency = clamp(windField.ambientB.x, 1e-6, 1.0);
    float lacunarity = clamp(windField.ambientB.z, 1.0, 4.0);
    float gain = clamp(windField.ambientB.w, 0.0, 0.95);
    float seed = clamp(windField.ambientC.z, -1000.0, 1000.0);
    float speed = max(windField.ambientA.w, 0.0);

    // World-space translation of the gust domain: the field has no mean flow,
    // so the only motion is a slow morph along a fixed diagonal (scaled by the
    // shared wind speed slider) plus the seed offset.
    vec3 worldOffset = vec3(0.7, 0.3, 1.0) * (time * 3.0 * speed)
                     + vec3(seed * 137.13, seed * 73.31, seed * 39.77);

    vec3 sum = vec3(0.0);
    float amp = 1.0;
    float ampSum = 0.0;
    float f = frequency;
    for (int i = 0; i < WIND_FBM_MAX_OCTAVES; ++i) {
        if (i >= octaves) break;
        // Every octave travels at the same world-space speed: its noise-space
        // point is scaled by its own frequency, offset included.
        vec3 q = p * f + worldOffset * f;
        sum.x += perlinNoise3D(q) * amp;
        sum.y += perlinNoise3D(q + kComponentOffsetY) * amp;
        sum.z += perlinNoise3D(q + kComponentOffsetZ) * amp;
        ampSum += amp;
        f *= lacunarity;
        amp *= gain;
    }
    return sum / max(ampSum, 1e-6);
}

// Wind velocity (m/s): the 3D fBm field scaled by ambientC.x. The direction is
// whatever the local fBm vector points at — nothing is authored. The vertical
// component is weighted by ambientC.y (1 = all three axes equal). NaN-safe:
// inputs clamped (huge world/time coordinates lose float precision and poison
// the hash otherwise).
vec3 windAmbient(vec3 p, float time) {
    // 0 = no ambient wind at all (skips the whole fBm; the branch is uniform).
    float gustAmplitude = clamp(windField.ambientC.x, 0.0, 128.0);
    if (gustAmplitude <= 0.0) return vec3(0.0);

    float t = clamp(time, 0.0, 86400.0);
    vec3 pc = vec3(clamp(p.x, -1e5, 1e5), clamp(p.y, -1e4, 1e4), clamp(p.z, -1e5, 1e5));
    vec3 gust = windFbm3(pc, t) * gustAmplitude;
    gust.y *= clamp(windField.ambientC.y, 0.0, 1.0);
    return gust;
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

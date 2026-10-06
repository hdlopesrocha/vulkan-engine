// Cheap procedural noise for SDF deformation. Hash-based (integer mixing, no
// textures, no sin/fract precision pitfalls, no vendor-specific behavior).
//
// Evaluation budget (documented contract for the raymarcher):
//   sdfNoise       = 8 hash evals (1 noise eval)
//   sdfFbm         = 3 noise evals
//   sdfDomainWarp  = 2 noise evals
//   sdfFlameDeform = 1 noise eval
// A warp + deform + one shape-noise sample costs at most 4 noise evals.

#ifndef SDF_NOISE_GLSL
#define SDF_NOISE_GLSL

uint sdfHashU(uvec3 u) {
    uint h = u.x * 374761393u + u.y * 668265263u + u.z * 1440662683u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    return h ^ (h >> 16u);
}

// Hash of a lattice point in [0, 1). Bit-mixing only, safe at large coords.
float sdfHash(vec3 p) {
    uvec3 u = floatBitsToUint(p);
    return float(sdfHashU(u)) * (1.0 / 4294967295.0);
}

// Value noise in [0, 1]: trilinear blend of the 8 corner hashes with a
// quintic fade (C2, hides cell-boundary creases in deformed SDFs).
float sdfNoise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    vec3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float n000 = sdfHash(i + vec3(0.0, 0.0, 0.0));
    float n100 = sdfHash(i + vec3(1.0, 0.0, 0.0));
    float n010 = sdfHash(i + vec3(0.0, 1.0, 0.0));
    float n110 = sdfHash(i + vec3(1.0, 1.0, 0.0));
    float n001 = sdfHash(i + vec3(0.0, 0.0, 1.0));
    float n101 = sdfHash(i + vec3(1.0, 0.0, 1.0));
    float n011 = sdfHash(i + vec3(0.0, 1.0, 1.0));
    float n111 = sdfHash(i + vec3(1.0, 1.0, 1.0));
    float nx00 = mix(n000, n100, u.x);
    float nx10 = mix(n010, n110, u.x);
    float nx01 = mix(n001, n101, u.x);
    float nx11 = mix(n011, n111, u.x);
    float nxy0 = mix(nx00, nx10, u.y);
    float nxy1 = mix(nx01, nx11, u.y);
    return mix(nxy0, nxy1, u.z);
}

// 3-octave fBm in [0, 1] (normalized by the amplitude sum).
float sdfFbm(vec3 p) {
    float total = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    float norm = 0.0;
    for (int i = 0; i < 3; i++) {
        total += sdfNoise(p * frequency) * amplitude;
        norm += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return total / max(norm, 1e-6);
}

// N-octave fBm in [0, 1] (normalized). sdfFbm(p) == sdfFbmOct(p, 3).
// Used by volumetric consumers (cloud bands) that need per-band detail.
float sdfFbmOct(vec3 p, int octaves) {
    float total = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    float norm = 0.0;
    for (int i = 0; i < 6; ++i) {
        if (i >= octaves) break;
        total += sdfNoise(p * frequency) * amplitude;
        norm += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return total / max(norm, 1e-6);
}

// Domain warp: offsets p by two low-frequency noise channels scaled by t
// (t = warp strength in world units). 2 noise evals.
vec3 sdfDomainWarp(vec3 p, float t, float seed) {
    float w1 = sdfNoise(p * 0.9 + vec3(seed * 13.1, seed * 5.7, -seed * 3.3));
    float w2 = sdfNoise(p * 1.7 - vec3(0.0, t * 0.15, 0.0) + vec3(seed * 7.7));
    return p + (vec3(w1, w2, (w1 + w2) * 0.5) - 0.5) * t;
}

// Rising-plume deformation: upward-scrolling noise plus a height-scaled
// lateral sway (1 noise eval + 1 sin). Returns a signed offset added to the
// SDF, roughly in [-turbulence, +turbulence]. seed decorrelates instances,
// turbulence scales the displacement, rise scrolls the noise upward over t.
float sdfFlameDeform(vec3 p, float t, float seed, float turbulence, float rise) {
    vec3 q = vec3(p.x * 1.6 + seed * 17.0,
                  p.y * 1.3 - t * rise,
                  p.z * 1.6 - seed * 11.0);
    float n = sdfNoise(q);
    float sway = sin(t * 2.1 + seed * 43.0 + p.y * 4.0) * 0.12;
    return ((n - 0.5) + sway) * turbulence;
}

// Spiky flame tongues: ridged value-noise displacement around the flame
// axis, stronger toward the tip. amp = 0 keeps the smooth rounded capsule;
// raise it for wavier, spikier flames. freq sets the tongue count around
// the axis. The ridged profile (sharp crests) reads as licking spikes;
// lowering amp "rounds up" the flame back to its smooth base shape.
// Returns a signed offset in [-amp, 0] (only grows the flame outward).
float sdfFlameSpikes(vec3 q, float h, float seed, float freq, float amp) {
    if (amp <= 1e-6 || h <= 1e-6) {
        return 0.0;
    }
    float ang = (dot(q.xz, q.xz) > 1e-12) ? atan(q.z, q.x) : 0.0;
    float hn = clamp(q.y / h, 0.0, 1.0);
    // Periodic around the axis (cos/sin pair) so there is no seam; the y
    // channel scrolls slowly upward for rising tongues.
    float n = sdfNoise(vec3(cos(ang), sin(ang), q.y * 0.35) * max(freq, 0.5)
                       + vec3(seed * 7.0, -seed * 3.0, seed * 11.0));
    float ridge = 1.0 - abs(2.0 * n - 1.0);
    ridge = ridge * ridge; // sharpen crests into spikes
    float w = mix(0.25, 1.0, hn); // calmer base, livelier tip
    return -amp * ridge * w;
}

// 4D value noise in [0, 1] (x, y, z + time): 16 corner hashes with a
// quintic fade. Used for time-evolving smoke bands — true temporal
// evolution, not just a drifting 3D domain. ~2x the hash cost of 3D, so it
// is reserved for the large/medium bands; fine detail stays 3D-advected.
uint sdfHashU4(uvec4 u) {
    uint h = u.x * 374761393u + u.y * 668265263u + u.z * 1440662683u + u.w * 2246822519u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    return h ^ (h >> 16u);
}

float sdfHash4(vec4 p) {
    uvec4 u = floatBitsToUint(p);
    return float(sdfHashU4(u)) * (1.0 / 4294967295.0);
}

float sdfNoise4(vec4 p) {
    vec4 i = floor(p);
    vec4 f = fract(p);
    vec4 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float c0000 = sdfHash4(i + vec4(0.0, 0.0, 0.0, 0.0));
    float c1000 = sdfHash4(i + vec4(1.0, 0.0, 0.0, 0.0));
    float c0100 = sdfHash4(i + vec4(0.0, 1.0, 0.0, 0.0));
    float c1100 = sdfHash4(i + vec4(1.0, 1.0, 0.0, 0.0));
    float c0010 = sdfHash4(i + vec4(0.0, 0.0, 1.0, 0.0));
    float c1010 = sdfHash4(i + vec4(1.0, 0.0, 1.0, 0.0));
    float c0110 = sdfHash4(i + vec4(0.0, 1.0, 1.0, 0.0));
    float c1110 = sdfHash4(i + vec4(1.0, 1.0, 1.0, 0.0));
    float c0001 = sdfHash4(i + vec4(0.0, 0.0, 0.0, 1.0));
    float c1001 = sdfHash4(i + vec4(1.0, 0.0, 0.0, 1.0));
    float c0101 = sdfHash4(i + vec4(0.0, 1.0, 0.0, 1.0));
    float c1101 = sdfHash4(i + vec4(1.0, 1.0, 0.0, 1.0));
    float c0011 = sdfHash4(i + vec4(0.0, 0.0, 1.0, 1.0));
    float c1011 = sdfHash4(i + vec4(1.0, 0.0, 1.0, 1.0));
    float c0111 = sdfHash4(i + vec4(0.0, 1.0, 1.0, 1.0));
    float c1111 = sdfHash4(i + vec4(1.0, 1.0, 1.0, 1.0));
    float nx000 = mix(c0000, c1000, u.x);
    float nx100 = mix(c0100, c1100, u.x);
    float nx010 = mix(c0010, c1010, u.x);
    float nx110 = mix(c0110, c1110, u.x);
    float nx001 = mix(c0001, c1001, u.x);
    float nx101 = mix(c0101, c1101, u.x);
    float nx011 = mix(c0011, c1011, u.x);
    float nx111 = mix(c0111, c1111, u.x);
    float nxy00 = mix(nx000, nx100, u.y);
    float nxy10 = mix(nx010, nx110, u.y);
    float nxy01 = mix(nx001, nx101, u.y);
    float nxy11 = mix(nx011, nx111, u.y);
    float nxyz0 = mix(nxy00, nxy10, u.z);
    float nxyz1 = mix(nxy01, nxy11, u.z);
    return mix(nxyz0, nxyz1, u.w);
}

// N-octave 4D fBm in [0, 1] (normalized). Frequency scales all four axes.
float sdfFbmOct4(vec4 p, int octaves) {
    float total = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    float norm = 0.0;
    for (int i = 0; i < 6; ++i) {
        if (i >= octaves) break;
        total += sdfNoise4(p * frequency) * amplitude;
        norm += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return total / max(norm, 1e-6);
}

#endif // SDF_NOISE_GLSL

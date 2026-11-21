// Gerstner wave train for the water surface: the wind-driven directional
// swell, its amplitude fade near the waterline, the horizontal pinch and the
// exact analytic normal of the resulting surface.
// The train's propagation DIRECTION and HEIGHT come from the shared 3D fBm
// wind field (sampled per water vertex by waterWindWave() in WaterTese.glsl):
// the caller passes the wind-blended direction and an amplitude already scaled
// by the saturating wind-sea response. This file is the pure wave geometry.
// Requires ubo.glsl (WaterParamsNamed) to be included first.
#ifndef GERSTNER_GLSL
#define GERSTNER_GLSL

#include "../../types/WaterParamsNamed.glsl"
#include "../../types/WaterWaveField.glsl"
#include "../noise/Perlin.glsl"

const int WATER_OCT_FULL = 64;
const int WATER_VERTEX_OCT = 2;



// ── ONE wind-driven Gerstner swell ───────────────────────────────────────
// A single directional swell travels along the wind-blended wave direction:
//   phase = k * dot(xz, dir) - omega * t + phase0
//   height = A * sin(phase)          (crest profile sharpened by the exponent)
//   xz    += Q * A * dir * cos(phase)  (the Gerstner pinch)
// The amplitude fades to zero over the last `shoreWaveFade` metres of water
// depth, so the wave approaches the shore and dies out there. In shallow water
// the CALLER bends `dir` toward the local shore direction (refraction), so the
// crests turn to follow the depth contours near the coast while running with
// the wind offshore. There is no spectrum: one swell with its analytic normal.
const int   WAVE_BANDS = 1;
// The Gerstner train is the SWELL alone: it is what the vertices are displaced
// with. The ripple detail is not geometry any more - it is an FBM normal
// perturbation evaluated per pixel in the fragment stage (see
// water_surface.glsl: "Gerstner for vertices, FBM for pixels").
const float WAVE_BAND_K[WAVE_BANDS]     = float[WAVE_BANDS](1.0);
const float WAVE_BAND_AMP[WAVE_BANDS]   = float[WAVE_BANDS](1.0);
const float WAVE_BAND_STEEP[WAVE_BANDS] = float[WAVE_BANDS](1.0);
const float WAVE_BAND_PHASE[WAVE_BANDS] = float[WAVE_BANDS](0.0);

WaterWaveField waterWaveField(vec3 xyz, float time, float depth, float amp,
                              vec2 waveDirIn, WaterParamsNamed wp, bool withFoam,
                              int octBudget, vec2 dPosdx, vec2 dPosdy) {
    WaterWaveField f;
    f.height = 0.0;
    f.grad = vec3(0.0);
    f.foam = 0.0;
    f.contact = 0.0;
    f.calmMask = 0.0;
    f.swell = 0.0;
    f.swellSlope = 0.0;
    f.breaking = 0.0;
    f.steepness = 0.0;
    f.disp = vec2(0.0);
    if (!wp.enableWaves || amp <= 0.0 || octBudget <= 0) return f;

    // Local water depth (m); a negative value is the deep sentinel.
    float d = (depth < 0.0) ? max(wp.shoreWaveFade, 1.0) * 4.0 : max(depth, 0.0);

    // Wave propagation direction: the caller passes the wind-driven direction
    // (shared fBm field, blended toward the shore in shallow water). The
    // authored angle is only the last-resort fallback.
    vec2 waveDir = (dot(waveDirIn, waveDirIn) > 1e-6)
        ? normalize(waveDirIn)
        : normalize(wp.waveDirection + vec2(1e-5, 0.0));

    const float TWO_PI = 6.28318530718;
    float kSw = TWO_PI * max(wp.wavePeriodScale, 1e-6);   // swell wavenumber
    float omSw = kSw * max(wp.waveSpeed, 0.0);            // swell angular frequency
    // Directional train: the phase runs along the wave direction, so the
    // crests are normal to the local wind and the train travels with it. The
    // caller's `amp` already carries the saturating wind-speed response.
    float phaseBase = dot(xyz.xz, waveDir);

    // Amplitude: the layer height (wind-scaled by the caller) and the fade to
    // nothing at the waterline.
    float shoreFade = (wp.shoreWaveFade > 0.0)
        ? smoothstep(0.0, wp.shoreWaveFade, d)
        : 1.0;
    float ampSwell = amp * max(wp.waveAmplitude, 0.0) * shoreFade;

    float h = 0.0;
    vec2 gParam = vec2(0.0);
    vec2 disp = vec2(0.0);
    mat2 jac = mat2(1.0);
    for (int i = 0; i < WAVE_BANDS; ++i) {
        float ki = kSw * WAVE_BAND_K[i];
        float omegai = omSw * WAVE_BAND_K[i];   // every band rides the swell
        // Analytic anti-aliasing: drop a band once its phase varies by more
        // than about half a cycle per pixel (the screen derivative of the
        // phase along the propagation direction).
        // Nyquist is pi radians per pixel. Window it at the actual limit
        // instead of fading early, so grazing views keep the swell slope.
        float phaseFw = ki * (abs(dot(dPosdx, waveDir)) + abs(dot(dPosdy, waveDir)));
        float aaFade = 1.0 - smoothstep(2.8, 5.6, phaseFw);
        if (aaFade <= 0.0) continue;

        float Ai = ampSwell * WAVE_BAND_AMP[i] * aaFade;
        float Qi = clamp(wp.waveSteepness * WAVE_BAND_STEEP[i], 0.0, 0.92);
        float phase = ki * phaseBase - omegai * time + WAVE_BAND_PHASE[i];
        float s = sin(phase);
        float c = cos(phase);
        // Crest sharpness: |sin|^e keeps the extremes and pulls the profile
        // toward them, so the crests (and troughs) narrow and the faces between
        // them steepen. e = 1 degenerates to the plain Gerstner profile.
        float sharp = max(wp.waveCrestSharpness, 1.0);
        float a = max(abs(s), 1e-5);
        float prof = sign(s) * pow(a, sharp);
        float dprof = sharp * pow(a, sharp - 1.0) * c;

        h += Ai * prof;
        gParam += Ai * ki * dprof * waveDir;
        // The Gerstner pinch: the surface point slides toward the crest, and
        // its derivative is what makes the analytic normal exact.
        disp += (Qi * Ai * c) * waveDir;
        jac += (-Qi * Ai * ki * s) * outerProduct(waveDir, waveDir);
        if (i == 0) {
            f.swell = prof;
            f.swellSlope = dprof;
        }
    }

    // Exact surface gradient of the Gerstner sum: J^-T * gParam (2x2 inverse).
    float det = jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0];
    det = (abs(det) < 1e-4) ? 1e-4 : det;
    vec2 g = vec2(jac[1][1] * gParam.x - jac[0][1] * gParam.y,
                  -jac[1][0] * gParam.x + jac[0][0] * gParam.y) / det;

    f.height = h;
    f.grad = vec3(g.x, 0.0, g.y);
    f.disp = disp;
    f.calmMask = 1.0;

    // Wind/pressure wave growth is handled by the caller (the amplitude is
    // already wind-scaled); no foam is generated by this geometry train.
    f.foam = 0.0;
    f.contact = 0.0;
    return f;
}

// Per-pixel FBM ripple detail for the shading normal.
//
// The vertex stages only carry the low-frequency Gerstner swell; this is the
// sub-tessellation roughness, so it exists on the shading normal alone. The
// tangent frame is built from the UNDISPLACED base normal (like the raster
// fragment stage) and the height gradient is expressed in world metres, which
// is what lets the normal be the one the surface would have with infinite
// tessellation. `detail` is the distance LOD factor; `waveDir` is the same
// wind-driven unit xz direction the Gerstner field used. Shared by the raster
// water surface and the octree ray-marched water path so both normals match.
vec3 waterRippleNormal(vec3 normal, vec3 baseN, vec3 basePos, vec2 waveDir,
                       float time, WaterParamsNamed wp, float detail) {
    vec3 n = normalize(normal);
    float rippleHeight = max(wp.rippleHeight, 0.0)
                       * clamp(wp.waveSteepness, 0.0, 1.0) * detail;
    if (!(rippleHeight > 0.0) || !(wp.noiseScale > 0.0)) return n;

    vec3 up = abs(baseN.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(up, baseN));
    vec3 B = cross(baseN, T);

    const int kRippleOctaves = 3;
    vec3 drift = vec3(waveDir.x, 0.0, waveDir.y) * (wp.waveSpeed * time);
    vec4 g = fbmGrad4D(vec4((basePos - drift) * wp.noiseScale,
                            time * wp.noiseTimeSpeed),
                       kRippleOctaves, wp.noisePersistence, wp.noiseLacunarity);
    vec3 rg = vec3(g.y, 0.0, g.w) * wp.noiseScale;   // d/dx, d/dz
    n = normalize(n - dot(rg, T) * rippleHeight * T
                    - dot(rg, B) * rippleHeight * B);
    // Same guard as the raster path: a collapsed direction here is what turned
    // the water black in the normal view.
    if (!(dot(n, n) > 0.25)) n = normalize(baseN);
    return n;
}

// Height-only entry point (debug views). The tessellation probe uses
// waterWaveTessProbe() below instead: it needs a density bias, not the full
// field.
float waterWaveDisplacement(vec3 xyz, float time, float depth, float amp,
                            vec2 shoreDir, WaterParamsNamed wp) {
    return waterWaveField(xyz, time, depth, amp, shoreDir, wp, false, WATER_OCT_FULL,
                          vec2(0.0), vec2(0.0)).height;
}

// Cheap tessellation probe: the same single sine the field displaces with
// (one evaluation, no noise chain). The mask is deliberately skipped - a
// triangle-density bias only needs the crest curvature - and the result is
// deterministic per shared edge, so both patches meeting at an edge agree.
// The range matches waterWaveDisplacement() (about [-1, 1]), so the density
// bias keeps its authored meaning.
// Cheap tessellation probe: the same single sine, deep-water phase.
float waterWaveTessProbe(vec3 pos, float time, WaterParamsNamed wp) {
    vec2 shoreDir = normalize(wp.waveDirection + vec2(1e-5, 0.0));
    float k = 6.28318530718 * max(wp.wavePeriodScale, 1e-6);
    return sin(k * dot(pos.xz, shoreDir) - k * max(wp.waveSpeed, 0.0) * time);
}

// Height + analytic gradient, vec4(height, dHeight/dx, dHeight/dy, dHeight/dz).
// The y component is 0: the field is a height field over xz.  This is the
// single wave field the TES displacement, the per-pixel normal and the
// caustic curvature all derive from. `octBudget` forwards the caller's
// evaluation-time LOD (WATER_OCT_FULL where none applies).
vec4 waterWaveSample(vec3 xyz, float time, float depth, float amp,
                     vec2 shoreDir, WaterParamsNamed wp, int octBudget,
                     vec2 dPosdx, vec2 dPosdy) {
    WaterWaveField f = waterWaveField(xyz, time, depth, amp, shoreDir, wp, false, octBudget,
                                      dPosdx, dPosdy);
    return vec4(f.height, f.grad);
}

// Second directional derivative of the wave height along `u`, for the
// wave-shape caustics (perf report 20 C3).
//
// The caustic term needs d²h/du². The original implementation central-
// differenced the analytic gradient over ±ec about the sun-ray entry point:
// TWO full field evaluations, i.e. 8 FBM chains and 32 four-dimensional Perlin
// evaluations per pixel, because the curvature spectrum needs every chain —
// the chop, the calm mask and both ridged trains all contribute, through the
// direct term and through the amplitude-modulation product rule.
//
// This helper keeps the same field and the same analytic gradient but
// differences it one-sidedly against an evaluation the shading normal has
// ALREADY paid for: `dhduCenter` is dot(waveField.grad, u) at the shaded
// surface point. The caustic therefore costs one extra evaluation instead of
// two. A one-sided difference of the analytic derivative has no cancellation
// (the gradient difference is O(step·h''), comparable to the gradients
// themselves) and its O(step) truncation is far inside the caustic softness
// clamp; the caller keeps its quarter-wavelength step, so the caustic detail
// still follows the band-limited field the surface is displaced with.
float waterWaveCurvature(vec3 pos, float time, float depth, float amp, vec2 shoreDir,
                         vec3 u, float step, float dhduCenter, WaterParamsNamed wp,
                         int octBudget, vec2 dPosdx, vec2 dPosdy) {
    WaterWaveField f = waterWaveField(pos + u * step, time, depth, amp, shoreDir, wp, false,
                                      octBudget, dPosdx, dPosdy);
    return (dot(f.grad, u) - dhduCenter) / step;
}

#endif // GERSTNER_GLSL

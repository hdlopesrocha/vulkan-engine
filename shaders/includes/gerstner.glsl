// Gerstner wave train for the water surface: the depth-derived shore
// distance, the swell plus its six ripple bands riding it, the horizontal
// pinch and the exact analytic normal of the resulting surface.
// Requires ubo.glsl (WaterParamsNamed) to be included first.
#ifndef GERSTNER_GLSL
#define GERSTNER_GLSL

const int WATER_OCT_FULL = 64;
const int WATER_VERTEX_OCT = 2;

struct WaterWaveField {
    float height;     // vertical displacement along the base normal
    vec3  grad;       // analytic d(height)/d(world position), y = 0 (height field)
    float foam;       // 0..1 whitewater coverage (breaking driven)
    float contact;    // 0..1 shoreline contact foam (water meets solid at depth 0)
    float calmMask;   // 1 when the field ran (waves active), 0 when it early-outed
    float swell;      // signed sine profile (the shore swell)
    float swellSlope; // its along-shore slope (the cosine)
    float breaking;   // unused (one sine, no breaking) - kept for the debug views
    float steepness;  // unused (one sine, no steepening) - kept for the debug views
    vec2  disp;       // Gerstner horizontal displacement (world xz) - applied by the vertex stages
};


// ── ONE shore sine ───────────────────────────────────────────────────────
// A single sine swell travels toward the shore along the local shore direction
// (the measured direction of decreasing water depth, falling back to the
// configured shore angle) and its amplitude fades to zero over the last
// `shoreWaveFade` metres of water depth: the wave approaches the shore and
// dies out there. There is no spectrum, no shoaling, no breaking and no region
// shaping - one sine on one water region. The only foam is the shoreline
// contact line.
// ONE swell with its ripples: a Gerstner train travelling toward the shore.
//
//   shoreDist = water depth / beach slope   (how far this point is from the shore)
//   phase_i   = k_i * shoreDist + omega_i * t
//   height    = sum  A_i * sin(phase_i)
//   xz       += sum  Q_i * A_i * shoreDir * cos(phase_i)     (the Gerstner pinch)
//
// Every band rides the SAME depth-derived phase, so the crests of the swell and
// of its ripples all lie on the depth contours - they follow the shoreline and
// bend with the bottom - and all of them travel in the shore direction
// together. The shading normal is the exact normal of the resulting Gerstner
// surface (the 2x2 Jacobian of the horizontal displacement is inverted), aimed
// along the local shore direction. `depth` is the measured water column; a
// negative value is the deep sentinel (no thickness signal).
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
                              vec2 shoreDirIn, WaterParamsNamed wp, bool withFoam,
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

    vec2 shoreDir = (dot(shoreDirIn, shoreDirIn) > 1e-6)
        ? normalize(shoreDirIn)
        : normalize(wp.waveDirection + vec2(1e-5, 0.0));

    const float TWO_PI = 6.28318530718;
    float kSw = TWO_PI * max(wp.wavePeriodScale, 1e-6);   // swell wavenumber
    float omSw = kSw * max(wp.waveSpeed, 0.0);            // swell angular frequency
    float shoreSlope = max(wp.shoreWaveSlope, 1e-4);
    float shoreDist = (depth >= 0.0) ? (d / shoreSlope) : dot(xyz.xz, shoreDir);

    // Amplitude: the layer height, the period-derived steepness scale, and the
    // fade to nothing at the waterline.
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
        // phase under the uniform beach-slope model).
        // Nyquist is pi radians per pixel. The old window (1.1 .. 2.4) faded the
        // band out well BEFORE that, and at grazing views the world footprint per
        // pixel is large enough that the whole swell lost its slope within a few
        // hundred metres - the normal went flat at range while looking correct up
        // close. Window it at the actual limit instead.
        float phaseFw = ki * (abs(dot(dPosdx, shoreDir)) + abs(dot(dPosdy, shoreDir)));
        float aaFade = 1.0 - smoothstep(2.8, 5.6, phaseFw);
        if (aaFade <= 0.0) continue;

        float Ai = ampSwell * WAVE_BAND_AMP[i] * aaFade;
        float Qi = clamp(wp.waveSteepness * WAVE_BAND_STEEP[i], 0.0, 0.92);
        float phase = ki * shoreDist + omegai * time + WAVE_BAND_PHASE[i];
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
        gParam += Ai * ki * dprof * shoreDir;
        // The Gerstner pinch: the surface point slides toward the crest, and
        // its derivative is what makes the analytic normal exact.
        disp += (Qi * Ai * c) * shoreDir;
        jac += (-Qi * Ai * ki * s) * outerProduct(shoreDir, shoreDir);
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

    // No foam: the surface is this one Gerstner train and nothing else.
    f.foam = 0.0;
    f.contact = 0.0;
    return f;
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

// Shared water noise + wave helpers.
// Requires `includes/perlin.glsl` and `includes/ubo.glsl` (WaterParamsGPU)
// to be included first.
//
// The water surface is built from ONE wave height field, waterWaveField(),
// exposed through waterWaveSample() (height + analytic gradient) and
// waterWaveDisplacement() (height only). The same field drives the TES
// displacement, the per-pixel analytic shading normal and the wave-shape
// caustics, so geometry, lighting and caustics can never disagree.

// Octave-budget sentinel for callers that must evaluate the full spectrum
// (any caller without its own LOD). The fragment stage passes its own budget
// instead — see waterWaveField()'s octBudget.
const int WATER_OCT_FULL = 64;

// Octave budget for the PER-VERTEX wave displacement (perf report 20 H6).
//
// The geometry can only represent features down to roughly its vertex spacing,
// so the finest octaves of the displacement are aliasing on the mesh: the
// per-layer spectrum's top octaves are sub-vertex-spacing on every chain
// (chop 0.25 m / 6 cm, cross train 2 m / 0.5 m, swell 8 m / 2 m, calm mask
// 16 m / 4 m, foam 1 m / 0.25 m), while the fragment stage re-derives the full
// spectrum per pixel for the shading normal anyway.
//
// This is ONE global cut applied to every vertex stage rather than a
// per-vertex spacing-derived budget, deliberately: the water pass measures its
// column from the distance between the FRONT and the BACK rasterised surface
// (waterBackDepthTex), so the two surfaces must be displaced by the SAME
// function or the measured column gains a wave-dependent term. A per-vertex
// budget (spacing- or distance-derived) differs between the two faces, which
// would inject metres of wave into the thickness where the water is thin and
// flip hasValidBackFace on and off.
//
// 2 of 4 octaves keeps ~80% of the linear spectrum amplitude (persistence 0.5)
// while removing half of the per-vertex noise cost in BOTH the geometry and
// the back-face pass. It is a single constant so it can be A/B'd cheaply.
const int WATER_VERTEX_OCT = 2;

// Minimum fine-detail fraction of the per-layer wave height.
//
// The shore-zone envelope (env) and the depth taper are the SWELL envelope:
// they legitimately damp the wave height in shallow water. Applied to the whole
// field they also delete the fine chop, and a height field with a zero gradient
// is a PERFECT MIRROR:
// the reflection stops following the wave normal and the water reads as a flat
// sheet with no movement. With the shipped zones (32/64/128 m) the envelope is
// below this floor for water shallower than ~24 m -- i.e. for any lake-scale
// water body.
//
// The CALM MASK is deliberately NOT covered by this floor: it is a complete
// gate (see waterWaveField), so a calm patch is exactly calm.
//
// This floor keeps the smallest fraction of the authored Wave Height alive, so
// the surface always has a live normal and the mirror always ripples. It is a
// fraction of the per-layer waveAmplitude, applied to the height AND the
// gradient together, so the geometry, the analytic normal and the two
// rasterised surfaces (front/back co-movement) all stay consistent. 0 restores
// the old behaviour (fully dead calm).

// Value/magnitude normalisation for every water FBM.
//
// fbm() divides by the sum of the octave amplitudes, but the Perlin gradients
// cancel across that sum, so the PRACTICAL range of the result is only about
// +/-0.43 (std 0.115) - not the +/-1 every parameter name implies. Measured
// consequences of reading it as if it were +/-1 / 0..1:
//
//   * sun glitter: smoothstep(threshold ~0.7, 1.0, noise) never fired at all
//     (the noise's 99.9th percentile is 0.32), so Glitter Intensity has never
//     produced a single sparkle;
//   * calm mask: m = noise*0.5+0.5 spans only 0.29..0.66, so with the shipped
//     default threshold 0.5 / softness 0.18 the mask could NEVER reach 1 (the
//     noise never gets to 0.68): every water pixel was damped and the waves
//     were fully removed over half the world. Lowering the threshold works
//     around the coverage, but the parameter semantics stay compressed until
//     the value is normalised;
//   * foam breakup: fn varied by +/-0.05 around 0.5, so Foam Noise Amount did
//     almost nothing.
//
// WAVE_FBM_GAIN maps the VALUE onto its nominal range. It is applied to the
// value only, NOT to the analytic gradient, deliberately: the gradient's
// per-octave weighting is what the surface shading was tuned against, so
// rescaling it would change the water's normal everywhere. The two are
// therefore not exact derivatives of each other any more - the value feeds the
// calm mask, the foam breakup, the amplitude/warp modulators and the debug
// views, the gradient feeds the normal and the caustics, and no consumer
// relates them.
const float WAVE_FBM_GAIN = 2.5;

float waterFbmNoise(vec3 xyz, float spatialScale, float time, float timeScale,
                    int octaves, float persistence, float lacunarity, vec3 offset) {
    return fbm(vec4((xyz + offset) * spatialScale, time * timeScale), octaves, persistence, lacunarity)
         * WAVE_FBM_GAIN;
}




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
const int   WAVE_BANDS = 7;
// Band 0 is the swell (Wave Period / Wave Height). Bands 1..6 are its ripples
// at a ~1.7x wavenumber step (a geometric progression, so the crests never
// beat into a regular pattern). Their amplitudes are graded so that every band
// carries a comparable SLOPE - which is what the eye reads as ripple detail -
// while its height falls off with the wavelength.
const float WAVE_BAND_K[WAVE_BANDS]     = float[WAVE_BANDS](1.0, 1.7, 2.9, 4.9, 8.3, 14.1, 24.0);
const float WAVE_BAND_AMP[WAVE_BANDS]   = float[WAVE_BANDS](1.0, 0.26, 0.12, 0.057, 0.027, 0.013, 0.0063);
// Gerstner pinch: only the swell and the first ripple displace the surface
// horizontally. The fine bands are pure shading detail - pinching them too
// would fold the Jacobian and break the normal they are meant to add.
const float WAVE_BAND_STEEP[WAVE_BANDS] = float[WAVE_BANDS](1.0, 0.5, 0.25, 0.0, 0.0, 0.0, 0.0);
const float WAVE_BAND_PHASE[WAVE_BANDS] = float[WAVE_BANDS](0.0, 1.9, 4.2, 2.7, 5.6, 0.8, 3.3);

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
        float phaseFw = ki * (abs(dot(dPosdx, shoreDir)) + abs(dot(dPosdy, shoreDir)));
        float aaFade = 1.0 - smoothstep(1.1, 2.4, phaseFw);
        if (aaFade <= 0.0) continue;

        float Ai = ampSwell * WAVE_BAND_AMP[i] * aaFade;
        float Qi = clamp(wp.waveSteepness * WAVE_BAND_STEEP[i], 0.0, 0.92);
        float phase = ki * shoreDist + omegai * time + WAVE_BAND_PHASE[i];
        float s = sin(phase);
        float c = cos(phase);

        h += Ai * s;
        gParam += Ai * ki * c * shoreDir;
        // The Gerstner pinch: the surface point slides toward the crest, and
        // its derivative is what makes the analytic normal exact.
        disp += (Qi * Ai * c) * shoreDir;
        jac += (-Qi * Ai * ki * s) * outerProduct(shoreDir, shoreDir);
        if (i == 0) {
            f.swell = s;
            f.swellSlope = c;
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

// Two-channel refraction distortion (perf report 20 C2).
//
// The previous implementation chained four 4D FBMs (noise1 4 oct + noise2 3 +
// noise3 2 + a separate 4-octave chain for nY) to produce a vec2: 13
// four-dimensional Perlin evaluations per pixel, ~1,664 PCG hashes, and the
// entire procedural cost of the default (waves-off) configuration. The
// distortion is a SCREEN-SPACE offset, so the fine octaves it paid for were
// sub-pixel at almost any camera distance.
//
// This version is two band-limited 3D FBMs: base scale matches the finest of
// the old layers (noiseScale * 0.30), the octave count is clamped to two, and
// the finest retained wavelength is therefore
//     wavelength = noisePeriod / (0.30 * lacunarity)
// (~3.3 m at the shipped defaults), which the caller uses to fade the whole
// distortion out once a pixel's world footprint exceeds it. The removed 4D
// time axis is replaced by a lateral world-space drift: a boiling pattern is
// not what water does, and the drift gives the same apparent motion the 4D
// axis did at a fraction of the cost. The 1.75 output scale preserves the old
// headroom (1 + 0.5 + 0.25) so refractionStrength keeps its authored meaning.
vec2 waterRefractionNoise(vec3 xyz, float noiseScale, float time,
                          int noiseOctaves, float noisePersistence, float noiseLacunarity) {
    const int kRefractionOctaves = 2;
    const float kRefractionAmplitude = 1.75;
    const vec3 kRefractionDrift = vec3(0.5, 0.0, 0.3); // m/s, lateral only
    int oct = clamp(noiseOctaves, 1, kRefractionOctaves);
    vec3 p = xyz + kRefractionDrift * time;
    float nX = fbm(p, oct, noisePersistence, noiseLacunarity);
    // Same decorrelating lattice offset the old nY chain used.
    float nY = fbm(p + vec3(50.0), oct, noisePersistence, noiseLacunarity);
    return vec2(nX, nY) * kRefractionAmplitude;
}

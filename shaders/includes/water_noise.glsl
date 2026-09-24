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

#include "gerstner.glsl"

float waterFbmNoise(vec3 xyz, float spatialScale, float time, float timeScale,
                    int octaves, float persistence, float lacunarity, vec3 offset) {
    return fbm(vec4((xyz + offset) * spatialScale, time * timeScale), octaves, persistence, lacunarity)
         * WAVE_FBM_GAIN;
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

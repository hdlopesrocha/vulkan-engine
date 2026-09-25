#pragma once
#include <glm/glm.hpp>

// Water rendering parameters (CPU-side)
struct WaterParams {
    // Deep-water phase speed of the primary swell (m/s). The angular
    // frequency is derived as frequency * speed, and the speed is reduced
    // with depth by waveShoalSpeed (shoaling).
    float waveSpeed = 6.0f;   // phase speed of the sine (m/s)
    // LEGACY (no effect, kept for layout/API stability): never uploaded to
    // the GPU — shaders use a neutral 1.0. Use Wave Height / Noise Scale.
    float waveScale = 0.03f;
    float refractionStrength = 1.0f;   // amount of Snell bending (0 = straight through)
    float fresnelPower = 5.0f;
    float transparency = 0.7f;
    float depthFalloff = 0.1f;

    // ── Water colour (ONE region) ────────────────────────────────────────
    // A single water tint. It fades to 0 at the waterline over
    // `tintShoreFadeDepth`, so the last water pixels near the shore are
    // transparent and show the bottom.
    glm::vec3 waterColor = glm::vec3(0.04f, 0.22f, 0.36f); // deep blue
    float tintShoreFadeDepth = 0.6f;
    int noiseOctaves = 4;
    // Persistence = 1/lacunarity keeps the SLOPE per octave constant, which is the
    // realistic gravity-wave falloff and spreads the surface detail across the
    // spectrum instead of concentrating it in the finest (sub-pixel) octave.
    float noisePersistence = 0.25f;
    float noiseLacunarity = 4.0f;
    // Noise feature PERIOD in world units (the shader converts to spatial
    // scale = 1 / period when packing to the GPU). Larger = broader features.
    float noisePeriod = 8.0f;
    float waterTint = 0.3f;
    float noiseTimeSpeed = 8.0f;

    // Refraction / absorption — per-layer water look. Single source of truth:
    // the former global RT duplicates (Water IOR, Absorption RGB/scale, Max
    // water thickness) were removed so these are tweaked here, per water layer.
    float ior = 1.333f;              // index of refraction for Snell air<->water
    glm::vec3 absorption = glm::vec3(0.35f, 0.12f, 0.08f); // Beer-Lambert RGB coefficients
    float absorptionScale = 0.3f;    // thickness multiplier for absorption
    float maxThickness = 64.0f;      // RT refraction hit clamp (m): deeper hits unresolved
    float shoreFadeDepth = 0.25f;    // water depth (m) over which the shoreline fades from fully transparent

    // Reflection / specular controls
    float reflectionStrength = 0.3f;  // How much reflection mixes into the surface [0..1]

    float specularIntensity = 2.0f;   // Brightness of specular highlight
    float specularPower = 128.0f;     // Sharpness of specular highlight
    float glitterIntensity = 1.5f;    // Brightness of sun glitter sparkles


    // Feature toggles
    bool enableReflection = true;
    bool enableRefraction = true;
    // ── Refraction/tint blur (final-pass, per material) ──
    // The final composite blurs the refraction + tint BODY with a disc whose
    // radius grows with the measured water depth (deeper water scatters more,
    // the shoreline stays crisp). Reflections and surface highlights are never
    // blurred. Per material, so one water layer can stay crisp while another
    // blurs. blurRadius is the radius clamp in pixels, blurDepthScale the
    // growth in pixels per meter of depth. Also gated globally by
    // Settings::blurEnabled: the blur runs only when both are on.
    bool enableBlur = true;
    float blurRadius = 8.0f;       // max blur radius (pixels)
    float blurDepthScale = 0.35f;  // blur growth (pixels per meter of depth)
    // If true, apply `reflectionStrength` uniformly across the surface
    // instead of modulating by Fresnel. Useful for debugging or stylized looks.
    bool uniformReflection = false;

    // Tessellation parameters (noise-adaptive water surface)
    // nearDist: camera distance at which tessellation reaches max level
    float tessNearDist = 128.0f;
    // farDist: camera distance at which tessellation drops to min level
    float tessFarDist = 1024.0f;
    float tessMinLevel = 1.0f;
    float tessMaxLevel = 16.0f;
    // [0..1] how much the bump/noise map influences tessellation.
    // 0 = purely distance-based, 1 = fully noise-adaptive.
    float tessNoiseInfluence = 0.3f;

    // ── Caustics (physical: refracted-sunlight focusing on the bottom) ──
    // The pattern is computed from the WAVE HEIGHT FIELD alone (the same
    // waterWaveSample() that displaces the surface): the bottom irradiance
    // is the inverse Jacobian of the refracted ray map,
    //   E/E0 = 1 / |1 + d·K·d²h/du²|,  K = cosθi/(n·cos³θt),
    // with u the sun azimuth, d the water column and h the wave height.
    // There is no separate caustic noise/scale/speed: the pattern rides the
    // waves and inherits their spectrum. Only the look controls below remain.
    glm::vec3 causticColor = glm::vec3(1.0f);   // tint of the focused sunlight
    float causticIntensity = 0.2f;              // strength of the added light
    // Softness: clamp floor on |J| (the 1/J fold is unbounded). Higher values
    // soften the bright ridges; 1.0 disables caustics (gain clamps to 1).
    float causticSoftness = 0.5f;

    // ── Shore-wave system (one sine swell, per water material) ────────────
    // The surface is a SINGLE sinusoidal swell travelling along
    // `shoreWaveAngle` (direction TOWARD the shore) with Wave Period and Wave
    // Speed. Its amplitude is gated by a sine band running along the shore
    // (Wave Mask Period): the swell arrives in sets, and a zero mask is a
    // fully calm band. There is no depth shaping and no ridged/choppy detail.
    // The zone depths below are read by the depth-region TINT and by the Foam
    // Band gate only - the wave itself is depth-independent.
    bool enableWaves = false;       // master toggle (layer 0 only by default)
    bool enableFoam = true;         // whitewater/foam rendering
    bool enableVolumetric = true;   // volumetric light scattering in the volume

    // Swell
    float wavePeriod = 512.0f;      // swell wavelength (world units)
    float waveAmplitude = 8.0f;

    // ── Shore wave: ONE sine ─────────────────────────────────────────────
    // A single sine swell travels toward the shore along the local shore
    // direction and its amplitude fades to zero over the last `shoreWaveFade`
    // metres of water depth, so it approaches the shore and dies out there.
    // No spectrum, no shoaling, no breaking, no regions.
    // Gerstner steepness: how far the crests pinch forward (0 = pure sine, the
    // crests stay symmetric; 1 = very sharp, forward-leaning crests).
    float waveSteepness = 0.7f;
    // Crest sharpness: exponent on the wave profile (|sin|^e). 1 = the plain
    // Gerstner profile, higher = narrower, spikier crests and troughs with
    // steeper faces between them.
    float waveCrestSharpness = 1.6f;
    float shoreWaveFade = 40.0f;    // depth (m) over which the swell fades out at the shore
    // Beach slope: metres of water depth per metre of horizontal distance from
    // the waterline. The wave phase runs on depth / slope (= the shore
    // distance), so this sets how far apart the crests are when they follow the
    // depth contours.
    float shoreWaveSlope = 0.02f;   // 1:50 beach

    // Propagation direction toward the shore (degrees in the XZ plane).
    // 0 = +Z, 90 = +X, 180 = -Z, 270 = -X. This is the FALLBACK direction:
    // the shader derives the local shore direction from the water-depth
    // gradient (toward thinning water) and only uses this angle where the
    // bottom cannot be measured. Set ShoreGradientStep = 0 to force it.
    float shoreWaveAngle = 0.0f;
    // Screen-texel step used to sample the water-depth gradient that yields
    // the shore direction. 0 = disable the gradient and use shoreWaveAngle.
    // A wider step keeps the deep-ocean direction stable (the bottom slope
    // there is small relative to the local ray spacing).
    float shoreGradientStep = 16.0f;   // texels: step of the depth-gradient shore solve


    // Foam (whitewater) look
    float foamCrestThreshold = 0.4f; // crest height where foam appears
    float foamTrailPhase = 1.0f;      // phase lag of the trailing foam band
    float foamDecay = 1.0f/1024.0f;          // foam extinction per meter below break
    float foamColorAmount = 1.0f;     // max foam color mix
    float foamNoisePeriod = 16.0f;    // texture feature period (world units)
    float foamNoiseSpeed = 0.15f;     // texture advection speed
    float foamNoiseAmount = 0.3f;     // how strongly noise breaks the foam
    // LEGACY (no effect, kept for layout/API stability): the persistent
    // shore-band foam line was removed when the foam was confined to the Foam
    // Band region - the Shore Line region is contact-only now. Use
    // foamContact* for the shoreline line.
    float foamShoreAmount = 1.0f;
    // Shoreline contact foam: the final line where the water meets the solid.
    // Peaks at zero depth and falls off over foamContactWidth; the fragment
    // stage forces the composite alpha up for it so the last visible water
    // pixels still render the line.
    float foamContactWidth = 24.0f;    // depth falloff band (world units)
    float foamContactAmount = 1.0f;   // line strength [0..1]
    float foamContactAlpha = 1.0f;   // minimum composite opacity of the line
    // The contact line arrives in waves: each crest pushes it up the shore and
    // it recedes between them. Residual strength kept between crests
    // (0 = fully retreats, 1 = continuous line).
    float foamContactFloor = 0.6f;
    float foamMaskFloor = 0.15f;     // foam left where the wave mask is 0 (calm patches)
    // Foam shape / motion:
    //  - edge: 0 = soft gradients, 1 = hard, well-defined foam edges
    //  - coverage: global foam coverage multiplier (lighter foam < 1)
    float foamEdge = 0.65f;
    float foamCoverage = 0.9f;
    float foamDiffuseFloor = 0.4f;    // ambient foam lighting floor
    float foamAmbient = 0.2f;        // constant ambient added to foam
    glm::vec3 foamColor = glm::vec3(1.0f);

    // Volumetric scattering (single-scattering approximation, sun-lit)
    float volumetricStrength = 0.15f; // in-scattered light amount
    float volumetricDensity = 0.08f;  // extinction per meter of water column
    float volumetricPhaseG = 0.4f;    // Henyey-Greenstein anisotropy [-0.95..0.95]
    glm::vec3 volumetricColor = glm::vec3(0.10f, 0.35f, 0.40f); // scatter tint
};

// Per-layer water look tier driven by the graphics-quality presets. Not
// packed to the GPU: the shaders gate on the existing per-feature fields
// (enableVolumetric/enableFoam/causticIntensity/glitterIntensity), which
// this helper sets.
enum class WaterQuality { Full, Minimal };

// Applies one tier to one layer:
//   Minimal — volumetric scattering, foam, caustics and glitter off.
//   Full    — those four fields back to their WaterParams{} struct defaults
//             (reset-to-defaults semantics: a per-layer authored override of
//             any of the four is not restored, it is reset). Colors,
//             amplitudes, waves and every other per-layer field are never
//             touched.
inline void applyWaterQuality(WaterParams& params, WaterQuality quality) {
    if (quality == WaterQuality::Minimal) {
        params.enableVolumetric = false;
        params.enableFoam = false;
        params.causticIntensity = 0.0f;
        params.glitterIntensity = 0.0f;
    } else {
        const WaterParams defaults{};
        params.enableVolumetric = defaults.enableVolumetric;
        params.enableFoam = defaults.enableFoam;
        params.causticIntensity = defaults.causticIntensity;
        params.glitterIntensity = defaults.glitterIntensity;
    }
}

// CPU mirror of shaders/includes/water_tint.glsl: ONE water region, so this is
// a single colour. `depth` is kept in the signature because the callers pass
// their depth signal in, but there is no depth-band palette any more.
inline glm::vec3 waterRegionTint(const WaterParams& p, float depth) {
    (void)depth;
    return p.waterColor;
}

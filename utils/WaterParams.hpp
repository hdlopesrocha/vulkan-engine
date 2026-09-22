#pragma once
#include <glm/glm.hpp>

// Water rendering parameters (CPU-side)
struct WaterParams {
    // Deep-water phase speed of the primary swell (m/s). The angular
    // frequency is derived as frequency * speed, and the speed is reduced
    // with depth by waveShoalSpeed (shoaling).
    float waveSpeed = 6.0f;
    // LEGACY (no effect, kept for layout/API stability): never uploaded to
    // the GPU — shaders use a neutral 1.0. Use Wave Height / Noise Scale.
    float waveScale = 0.03f;
    float refractionStrength = 0.03f;
    float fresnelPower = 5.0f;
    float transparency = 0.7f;
    float depthFalloff = 0.1f;

    // ── Depth-region tint ───────────────────────────────────────────────
    // The water tint is a 5-stop color ramp keyed to the measured water
    // DEPTH and the shore-wave zone boundaries (zoneShallowDepth /
    // zoneBreakDepth / zoneDeepDepth): the tint color at a pixel is the
    // region color of its depth band, smoothly blended across the
    // boundaries. Region order (shallow → deep): shore line, foam decay
    // band, breaker line, shoaling band, open ocean.
    glm::vec3 regionShoreColor = glm::vec3(1.0f, 0.0f, 0.0f);   // d < zoneShallowDepth
    glm::vec3 regionShallowColor = glm::vec3(1.0f, 1.0f, 0.0f); // foam decay band
    glm::vec3 regionBreakerColor = glm::vec3(0.0f, 1.0f, 0.0f); // breaker line
    glm::vec3 regionShoalColor = glm::vec3(0.03f, 1.0f, 1.0f);   // shoaling band
    glm::vec3 regionDeepColor = glm::vec3(0.0f, 0.0f, 1.0f);     // open ocean
    // Boundary blend half-width as a fraction [0..0.5] of the adjacent zone
    // spans: 0 = hard region edges, 0.5 = fully soft ramp.
    float regionBlendSoftness = 0.35f;
    // Water depth (m) over which the tint fades to 0 at the waterline, so
    // the last water pixels near the shore are transparent and show the
    // bottom with no tint. 0 = disable the tint shoreline fade.
    float tintShoreFadeDepth = 0.6f;
    int noiseOctaves = 4;
    float noisePersistence = 0.5f;
    float noiseLacunarity = 4.0f;
    // Noise feature PERIOD in world units (the shader converts to spatial
    // scale = 1 / period when packing to the GPU). Larger = broader features.
    float noisePeriod = 4.0f;
    float waterTint = 0.3f;
    float noiseTimeSpeed = 1.0f;

    // Refraction / absorption — per-layer water look. Single source of truth:
    // the former global RT duplicates (Water IOR, Absorption RGB/scale, Max
    // water thickness) were removed so these are tweaked here, per water layer.
    float ior = 1.333f;              // index of refraction for Snell air<->water
    glm::vec3 absorption = glm::vec3(0.35f, 0.12f, 0.08f); // Beer-Lambert RGB coefficients
    float absorptionScale = 1.0f;    // thickness multiplier for absorption
    float maxThickness = 6.0f;       // clamp for RT hit thickness (kills far-hit blackouts)
    float shoreFadeDepth = 0.25f;    // water depth (m) over which the shoreline fades from fully transparent

    // Reflection / specular controls
    float reflectionStrength = 0.6f;  // How much reflection mixes into the surface [0..1]
    float specularIntensity = 2.0f;   // Brightness of specular highlight
    float specularPower = 128.0f;     // Sharpness of specular highlight
    float glitterIntensity = 1.5f;    // Brightness of sun glitter sparkles

    // Vertical bump amplitude for water geometry
    float bumpAmplitude = 8.0f;

    // Feature toggles
    bool enableReflection = true;
    bool enableRefraction = true;
    // ── Refraction/tint blur (final-pass, per material) ──
    // The final composite blurs the refraction + tint BODY with a disc whose
    // radius grows with the measured water depth (deeper water scatters more,
    // the shoreline stays crisp). Reflections and surface highlights are never
    // blurred. Per material, so one water layer can stay crisp while another
    // blurs. blurRadius is the radius clamp in pixels, blurDepthScale the
    // growth in pixels per meter of depth.
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

    // ── Shore-wave system (thickness-zoned, per water material) ──────────
    // Waves travel along `shoreWaveAngle` (direction TOWARD the shore) and
    // are shaped by configurable water-thickness zones:
    //   d >= zoneDeepDepth                 : open ocean, sharp big swell
    //   zoneBreakDepth <= d < zoneDeepDepth: shoaling band (gains height,
    //                                        sharpens, whitecaps appear)
    //   d ~ zoneBreakDepth                 : breaker line, foam is born
    //   zoneShallowDepth <= d < zoneBreak  : foam rides shoreward and fades
    //   d < zoneShallowDepth               : residual line wave -> 0 at shore
    // Only the waves are gated by enableWaves; foam follows its own toggle.
    bool enableWaves = false;       // master toggle (layer 0 only by default)
    bool enableFoam = true;         // whitewater/foam rendering
    bool enableVolumetric = true;   // volumetric light scattering in the volume

    // Zone boundaries (water thickness, world units)
    float zoneDeepDepth = 128.0f;   // >= : open-ocean swell
    float zoneBreakDepth = 64.0f;   // breaker line (crash + foam birth)
    float zoneShallowDepth = 32.0f; // below : line wave decaying to 0 at shore

    // Swell shape
    float wavePeriod = 128.0f;      // primary swell wavelength (world units)
    float waveSharpDeep = 2.5f;     // crest sharpness in deep water
    float waveSharpBreak = 4.5f;    // crest sharpness at the break line
    float waveSharpShallow = 1.5f;  // crest sharpness in the shore band
    float waveShoalGain = 0.8f;     // amplitude gain from deep -> break depth
    float waveShoalSpeed = 0.6f;    // celerity drop with depth [0..1]
    float waveShallowDecay = 1.5f;  // amplitude decay exponent, break -> shallow
    float waveLineAmplitude = 0.25f;// residual line-wave height fraction
    // Global depth taper of the wave HEIGHT for every component (trains, chop
    // and curl): pow(clamp(depth / zoneDeepDepth, 0, 1), falloff) so the
    // height falls monotonically from the deep zone down to 0 at the shore.
    // 0 = disabled (the zone envelope alone shapes the height).
    float waveHeightFalloff = 1.0f;
    float breakerAmplitude = 0.6f;  // extra crest height at the break line
    float breakerWidth = 10.0f;     // depth half-width of the breaker bump
    // Curly plunging-breaker shape (active only where the wave is breaking):
    //  - curl: forward-leaning profile skew that steepens the lip;
    //    sign flips the lean direction
    //  - crestCurve: hooks the breaking crest line into a curl
    float breakerCurl = 0.6f;       // lip skew [-0.9..0.9]
    float breakerCrestCurve = 0.4f; // crest-line hook (0 = straight crests)
    float waveChopAmount = 0.25f;   // FBM chop mixed into the directional swell
    float whitecapOnset = 0.15f;    // shoal progress where whitecaps start [0..1]
    // Organic variation of the sharp crests. The crests themselves are ridged
    // Perlin multifractal (see the shader); these control the extra noise
    // modulation (uses the same FBM spectrum as the chop):
    //  - phase warp: domain-warp drift of the ridged crests (feature units,
    //    1.0 = one ridge-feature shift)
    //  - amplitude variation: local crest height variation [0..1]
    //  - ridge stretch: along/across frequency ratio; higher = longer crest
    //    lines (more wave-like, less blob-like)
    float waveWarpAmount = 0.5f;    // crest phase warp (feature units)
    float waveAmpVariation = 0.5f;  // local crest amplitude variation [0..1]
    float waveRidgeStretch = 4.0f;  // ridged-crest anisotropy (along/across)

    // Second shoreward train (different scale/speed; breaks up the crest
    // lines). It reuses the SAME shore movement as the primary train — the
    // phase offset just shifts it along the shore direction.
    float crossWavePeriod = 32.0f; // cross train wavelength (world units)
    float crossWaveSpeed = 4.5f;
    float crossWaveAmplitude = 0.4f;
    float crossWavePhase = 0.0f;    // offset along the shore direction (world units)

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
    float shoreGradientStep = 16.0f;

    // Organic amplitude mask: low-frequency noise that can remove waves
    // entirely in patches (some places stay calm).
    float waveMaskPeriod = 256.0f;
    float waveMaskThreshold = 0.5f;
    float waveMaskSoftness = 0.18f;
    float waveMaskSpeed = 0.05f;

    // Foam (whitewater) look
    float foamCrestThreshold = 0.4f; // crest height where foam appears
    float foamTrailPhase = 1.0f;      // phase lag of the trailing foam band
    float foamDecay = 1.0f/1024.0f;          // foam extinction per meter below break
    float foamColorAmount = 1.0f;     // max foam color mix
    float foamNoisePeriod = 16.0f;    // texture feature period (world units)
    float foamNoiseSpeed = 0.15f;     // texture advection speed
    float foamNoiseAmount = 0.3f;     // how strongly noise breaks the foam
    float foamShoreAmount = 1.0f;     // persistent foam line near the shore
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
    //  - shoreSpeed: foam advection speed factor at the shoreline (1 at the
    //    breaker): foam races off the curl, then slows as it nears the shore
    //  - lagGrowth: how much the trailing foam falls behind the lip as the
    //    wave approaches the shore
    float foamEdge = 0.65f;
    float foamCoverage = 0.9f;
    float foamShoreSpeed = 0.25f;
    float foamLagGrowth = 1.5f;
    float foamDiffuseFloor = 0.4f;    // ambient foam lighting floor
    float foamAmbient = 0.2f;        // constant ambient added to foam
    glm::vec3 foamColor = glm::vec3(1.0f);

    // Volumetric scattering (single-scattering approximation, sun-lit)
    float volumetricStrength = 0.15f; // in-scattered light amount
    float volumetricDensity = 0.08f;  // extinction per meter of water column
    float volumetricPhaseG = 0.4f;    // Henyey-Greenstein anisotropy [-0.95..0.95]
    glm::vec3 volumetricColor = glm::vec3(0.10f, 0.35f, 0.40f); // scatter tint
};

// CPU mirror of shaders/includes/water_tint.glsl: the 5-stop depth-region
// tint ramp (shore → foam band → breaker line → shoaling → deep ocean), keyed
// to the same wave-zone boundaries with the same soft blending. Used where the
// CPU needs a representative water tint (e.g. the RT proxy water albedo).
inline glm::vec3 waterRegionTint(const WaterParams& p, float depth) {
    const float zDeep = glm::max(p.zoneDeepDepth, 1.0f);
    const float zBreak = glm::clamp(p.zoneBreakDepth, 0.001f, zDeep);
    const float zShallow = glm::clamp(p.zoneShallowDepth, 0.0f, zBreak);
    const float soft = glm::clamp(p.regionBlendSoftness, 0.0f, 0.5f);

    const float s1 = zShallow;
    const float s2 = zBreak;
    const float s3 = 0.5f * (zBreak + zDeep);
    const float s4 = zDeep;

    const float w1 = soft * glm::max(s1, 0.001f);
    const float w2 = soft * glm::max(glm::min(s2 - s1, s3 - s2), 0.001f);
    const float w3 = soft * glm::max(glm::min(s3 - s2, s4 - s3), 0.001f);
    const float w4 = soft * glm::max(s4 - s3, 0.001f);

    glm::vec3 c = p.regionShoreColor;
    c = glm::mix(c, p.regionShallowColor, glm::smoothstep(glm::max(s1 - w1, 0.0f), s1 + w1, depth));
    c = glm::mix(c, p.regionBreakerColor, glm::smoothstep(s2 - w2, s2 + w2, depth));
    c = glm::mix(c, p.regionShoalColor, glm::smoothstep(s3 - w3, s3 + w3, depth));
    c = glm::mix(c, p.regionDeepColor, glm::smoothstep(s4 - w4, s4 + w4, depth));
    return c;
}

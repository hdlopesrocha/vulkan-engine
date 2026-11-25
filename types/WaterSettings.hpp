#pragma once
#include <glm/glm.hpp>

// Water rendering parameters (CPU-side)
struct WaterSettings {

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
    float noisePeriod = 128.0f;
    float waterTint = 0.3f;
    float noiseTimeSpeed = 8.0f;

    // Refraction / absorption — per-layer water look. Single source of truth
    // for the water IOR, absorption RGB/scale and maximum water thickness.
    float ior = 1.333f;              // index of refraction for Snell air<->water
    glm::vec3 absorption = glm::vec3(0.35f, 0.12f, 0.08f); // Beer-Lambert RGB coefficients
    float absorptionScale = 0.3f;    // thickness multiplier for absorption
    float maxThickness = 64.0f;      // refraction march clamp (m): deeper hits unresolved
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

    // ── Wave system (one wind-driven Gerstner swell, per water material) ──
    // The wave DIRECTION and HEIGHT are driven by the shared 3D fBm wind field
    // (the SAME field the vegetation/clouds/fire sample): the train travels
    // with the local wind in deep water, bends toward the shore in shallow
    // water (refraction), and its height grows with the wind speed up to the
    // wind response reference (see the Wind Drive block below). Wave Period
    // sets the crest spacing along the propagation direction and Wave Speed
    // the phase speed.
    bool enableWaves = false;       // master toggle (layer 0 only by default)
    bool enableFoam = true;         // whitewater/foam rendering
    bool enableVolumetric = true;   // volumetric light scattering in the volume

    // Swell
    float waveSpeed = 32.0f;   // phase speed of the sine (m/s)
    float wavePeriod = 512.0f;      // swell wavelength (world units)
    float waveAmplitude = 8.0f;

    // ── Shore wave: ONE sine ─────────────────────────────────────────────
    // A single sine swell travels toward the shore along the local shore
    // direction and its amplitude fades to zero over the last `shoreWaveFade`
    // metres of water depth, so it approaches the shore and dies out there.
    // No spectrum, no shoaling, no breaking, no regions.
    // Gerstner steepness: how far the crests pinch forward (0 = pure sine, the
    // crests stay symmetric; 1 = very sharp, forward-leaning crests).
    float waveSteepness = 0.9f;
    // Crest sharpness: exponent on the wave profile (|sin|^e). 1 = the plain
    // Gerstner profile, higher = narrower, spikier crests and troughs with
    // steeper faces between them.
    float waveCrestSharpness = 1.6f;
    // Ripple HEIGHT in metres (trough-to-crest at full Wave Steepness). The
    // ripples are a height field, not just a normal perturbation: the shading
    // normal is derived from this height, i.e. the normal the surface would
    // have if it were displaced by the ripples at infinite tessellation.
    float rippleHeight = 2.0f;
    float shoreWaveFade = 40.0f;    // depth (m) over which the swell fades out at the shore
    // Unused by the wave geometry since the phase became directional (the
    // train follows the wind, not the depth contours). Kept in the settings /
    // GPU layout for compatibility; the wave direction comes from the shared
    // wind field and the shallow blend below.
    float shoreWaveSlope = 0.02f;   // legacy (1:50 beach)

    // Propagation direction fallback (degrees in the XZ plane).
    // 0 = +Z, 90 = +X, 180 = -Z, 270 = -X. Used where the shared wind is calm
    // and the local shore direction cannot be measured; also the target of the
    // shallow-water refraction blend.
    float shoreWaveAngle = 0.0f;
    // Screen-texel step used to sample the water-depth gradient that yields
    // the shore direction. 0 = disable the gradient and use shoreWaveAngle.
    // A wider step keeps the deep-ocean direction stable (the bottom slope
    // there is small relative to the local ray spacing).
    float shoreGradientStep = 16.0f;   // texels: step of the depth-gradient shore solve

    // ── Wind-driven waves (shared 3D fBm wind field) ─────────────────────
    // The wave train direction and height come from the SAME wind field the
    // vegetation/clouds/fire sample (set=0 binding 27): waves travel with the
    // local wind in deep water, bend toward the shore in shallow water
    // (refraction), and their height grows with the wind speed up to
    // `windWaveReference` (saturating wind-sea response).
    //   0 = the authored height stays constant (the wind only steers the
    //       direction), 1 = fully wind-driven height.
    float windWaveResponse = 0.85f;
    // Wind speed (m/s) reaching ~63% of the response (calm air ~ flat water).
    float windWaveReference = 10.0f;
    // Water depth (m) over which the wave direction blends from the wind to
    // the shore direction. 0 = derive it from shoreWaveFade.
    float windWaveShallowBlend = 0.0f;


    // Foam (whitewater) look
    float foamCrestThreshold = 0.4f; // crest height where foam appears
    float foamTrailPhase = 1.0f;      // phase lag of the trailing foam band
    float foamDecay = 1.0f/1024.0f;          // foam extinction per meter below break
    float foamColorAmount = 1.0f;     // max foam color mix
    float foamNoisePeriod = 16.0f;    // texture feature period (world units)
    float foamNoiseSpeed = 0.15f;     // texture advection speed
    float foamNoiseAmount = 0.3f;     // how strongly noise breaks the foam
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

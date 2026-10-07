#ifndef WATER_PARAMS_NAMED_GLSL
#define WATER_PARAMS_NAMED_GLSL

// Extracted from shaders/includes/SceneBindings.glsl (single-struct GLSL type).

// ── Named view over the packed WaterParams. ───────────────────────────
// Same data, descriptive names: the packed wave-mask threshold reads as
// `wp.waveMaskThreshold`, the foam-noise period scale as
// `wp.foamNoisePeriodScale`, and so on for every component. Built once where the params are obtained (the
// SSBO array itself stays packed, so the layout is untouched), then passed to
// the helpers BY VALUE - the compiler keeps it in registers and drops every
// field a call site does not read.
//
// Fields ending in `Scale` are spatial scales, i.e. 1/period: the CPU converts
// the authored periods once at the upload boundary (waterGpuPeriodsToScales).
struct WaterParamsNamed {
    float refractionStrength;
    float fresnelPower;
    float transparency;
    float reflectionStrength;
    float waterTint;
    float noiseScale;
    float noiseOctaves;
    float noisePersistence;
    float noiseTimeSpeed;
    float noiseLacunarity;
    float specularIntensity;
    float specularPower;
    float glitterIntensity;
    bool enableBlur;
    float blurRadius;
    float blurDepthScale;
    float tessNoiseInfluence;
    float bumpAmplitude;
    float depthFalloff;
    bool enableReflection;
    bool enableRefraction;
    bool uniformReflection;
    float tessNearDist;
    float tessFarDist;
    float tessMinLevel;
    float tessMaxLevel;
    vec3 causticColor;
    float causticSoftness;
    float causticIntensity;
    vec3 absorption;
    float absorptionScale;
    float waterIor;
    float maxThickness;
    float shoreFadeDepth;
    bool enableWaves;
    bool enableFoam;
    bool enableVolumetric;
    vec2 waveDirection;
    float breakerWidth;
    float wavePeriodScale;
    float shoreWaveFade;
    float shoreWaveSlope;
    float waveSteepness;
    float waveCrestSharpness;
    float rippleHeight;
    vec3 waterColor;
    float waveSpeed;
    float waveAmplitude;
    float shoreGradientStep;
    float foamCrestThreshold;
    float foamTrailPhase;
    float foamDecay;
    float foamColorAmount;
    float foamNoisePeriodScale;
    float foamNoiseSpeed;
    float foamNoiseAmount;
    float foamShoreAmount;
    float foamMaskFloor;
    float foamDiffuseFloor;
    float foamAmbient;
    float foamContactWidth;
    float foamContactAmount;
    float foamContactAlpha;
    float foamContactFloor;
    float foamEdge;
    float foamCoverage;
    vec3 foamColor;
    float volumetricStrength;
    float volumetricDensity;
    float volumetricPhaseG;
    vec3 volumetricColor;
    float tintShoreFadeDepth;
    // Music-reactive audio input (MusicWidget analysis, per-frame upload).
    float audioAmplitude;
    float bassEnergy;
    float midEnergy;
    float highEnergy;
    float beatIntensity;
    bool musicReactive;
};

#endif // WATER_PARAMS_NAMED_GLSL

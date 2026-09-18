#pragma once
#include <glm/glm.hpp>

// Water rendering parameters (CPU-side)
struct WaterParams {
    float waveSpeed = 0.5f;
    // LEGACY (no effect, kept for layout/API stability): never uploaded to
    // the GPU — shaders use a neutral 1.0. Use Wave Height / Noise Scale.
    float waveScale = 0.03f;
    float refractionStrength = 0.03f;
    float fresnelPower = 5.0f;
    float transparency = 0.7f;
    glm::vec3 shallowColor = glm::vec3(0.1f, 0.4f, 0.5f);
    glm::vec3 deepColor = glm::vec3(0.0f, 0.15f, 0.25f);
    float depthFalloff = 0.1f;
    int noiseOctaves = 4;
    float noisePersistence = 0.5f;
    float noiseLacunarity = 2.0f;
    float noiseScale = 0.4f;
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

    // Depth-based wave attenuation: distance (world units) over which waves
    // transition from zero displacement (at solid surface) to full amplitude.
    // 0 = disabled (no depth-based attenuation).
    float waveDepthTransition = 20.0f;

    // Feature toggles
    bool enableReflection = true;
    bool enableRefraction = true;
    // LEGACY (no effect): the PCF scene-color blur was removed when the water
    // pass was decoupled from the solid pass (no scene texture to blur).
    bool enableBlur = true;
    // If true, apply `reflectionStrength` uniformly across the surface
    // instead of modulating by Fresnel. Useful for debugging or stylized looks.
    bool uniformReflection = false;

    // LEGACY (no effect, see enableBlur): blur kernel parameters.
    float blurRadius = 8.0f;    // texel radius of blur kernel
    int   blurSamples = 4;      // number of blur taps per axis (NxN kernel)

    // Volume depth-based effect transitions
    // LEGACY (no effect, see enableBlur): blur has no implementation.
    float volumeBlurRate = 0.004f;   // exponential rate: blur ramps with water thickness
    float volumeBumpRate = 0.05f;  // exponential rate: bump ramps with water thickness
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
    // Depth reference (world units) used by the water-tint volume ramp.
    float causticDepthScale = 128.0f;
};

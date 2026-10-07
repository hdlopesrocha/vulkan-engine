#pragma once
#include <cstddef>
#include <glm/glm.hpp>

// GPU-side water params UBO (canonical shared layout, mirrors the GLSL
// WaterParams in shaders/ssbo/WaterParams.glsl; the SSBO array stride is
// sizeof(this)). Packed from the CPU WaterSettings by makeWaterParams.
struct WaterParams {
    glm::vec4 params1;  // x=refractionStrength, y=fresnelPower, z=transparency, w=reflectionStrength
    glm::vec4 params2;  // x=waterTint, y=noise period (converted to scale at upload), z=noiseOctaves, w=noisePersistence
    glm::vec4 params3;  // x=noiseTimeSpeed, y=noiseLacunarity, z=specularIntensity, w=specularPower
    glm::vec4 glitterParams; // x=glitterIntensity, yzw=unused
    glm::vec4 blurParams; // x=enableBlur, y=max radius (pixels), z=radius per meter (px/m), w=unused
    glm::vec4 waveParams; // x=tessNoiseInfluence, y=unused, z=waveAmplitude, w=depthFalloff
    glm::vec4 reserved1;  // x=enableReflection, y=enableRefraction, zw=unused
    glm::vec4 reserved2;  // w=uniformReflection, xyz=unused
    glm::vec4 reserved3;  // unused (legacy cubemap-available flag removed with Solid360)
    glm::vec4 tessParams; // x=tessNearDist, y=tessFarDist, z=tessMinLevel, w=tessMaxLevel
    glm::vec4 causticColor; // xyz = color of the focused sunlight, w = unused
    glm::vec4 causticParams; // x=softness (|J| floor), y=intensity, zw=unused
    glm::vec4 causticExtraParams; // reserved (wave-shape caustics need no mode/line/speed knobs)
    glm::vec4 absorptionParams; // xyz = Beer-Lambert coeff, w = absorption scale
    glm::vec4 refractionParams; // x=IOR, y=max thickness cap, z=shore fade depth, w=unused

    // ── Shore-wave system (all zones/thresholds configurable, no shader
    //    magic numbers; mirrors the GLSL WaterParams struct). ──
    glm::vec4 waveToggles;      // x=enableWaves, y=enableFoam, z=enableVolumetric, w=unused
    glm::vec4 waveZones;        // x=deep depth(>=), y=break depth, z=shallow depth, w=unused
    glm::vec4 waveDirection;    // xy=shore direction fallback (unit, world XZ), zw=unused
    glm::vec4 waveShape;        // x=shoreWaveSlope (m depth per m), y=waveCrestSharpness, zw=unused
    glm::vec4 waveShoal;        // z=waveSteepness (Gerstner pinch), xyw=unused
    glm::vec4 waveComponent1;   // x=dominant swell period (scale at upload), y=dispersion scale, z=height scale, w=unused
    glm::vec4 waveComponent2;   // reserved
    glm::vec4 waveBreaker;      // reserved
    glm::vec4 waveCurl;         // reserved
    glm::vec4 waveWarp;         // xyz reserved, w=shore gradient step(texels)
    glm::vec4 waveMask;         // x=shoreWaveFade (depth over which the swell fades out at the shore)
    glm::vec4 foamParams;       // x=crest threshold, y=trail phase, z=decay/m, w=color amount
    glm::vec4 foamNoise;        // x=foam noise period (converted to scale at upload), y=time speed, z=noise amount, w=shore amount
    glm::vec4 foamExtra;        // x=mask floor, y=diffuse floor, z=ambient, w=unused
    glm::vec4 foamContact;      // x=contact width(world), y=contact amount, z=contact alpha, w=contact pulse floor
    glm::vec4 foamShape;        // x=edge hardness, y=coverage, zw=reserved
    glm::vec4 foamColor;        // rgb=foam color, a=unused
    glm::vec4 volumetricParams; // x=strength, y=density, z=Henyey-Greenstein g, w=unused
    glm::vec4 volumetricColor;  // rgb=volumetric scatter tint, a=unused

    // ── Depth-region tint (mirrors utils/WaterSettings.hpp) ──
    glm::vec4 regionShoreColor;   // rgb = the single water colour (one region)
    glm::vec4 regionShallowColor; // reserved
    glm::vec4 regionBreakerColor; // reserved
    glm::vec4 regionShoalColor;   // reserved
    glm::vec4 regionDeepColor;    // reserved
    glm::vec4 regionTintParams;   // x=unused, y=tint shore fade depth (m), zw=unused

    // ── Music-reactive audio input (written by WaterRenderer::updateMusicAudio,
    //    fed live by the MusicWidget analysis; read by MUSIC_REACTIVE_WATER) ──
    glm::vec4 musicAudio1; // x=smoothed audioAmplitude 0..1, y=bassEnergy 0..1, z=midEnergy 0..1, w=highEnergy 0..1
    glm::vec4 musicAudio2; // x=beatIntensity 0..1, y=music-reactive input enabled 1/0, zw=reserved
};

// The GLSL mirror (shaders/ssbo/WaterParams.glsl) must declare the exact same
// vec4 sequence; the SSBO array stride is sizeof(this).
static_assert(sizeof(WaterParams) == 42 * sizeof(glm::vec4),
              "WaterParams layout drifted from shaders/ssbo/WaterParams.glsl");
static_assert(offsetof(WaterParams, params1) == 0, "WaterParams.params1 offset");
static_assert(offsetof(WaterParams, musicAudio1) == 40 * sizeof(glm::vec4), "WaterParams.musicAudio1 offset");
static_assert(offsetof(WaterParams, musicAudio2) == 41 * sizeof(glm::vec4), "WaterParams.musicAudio2 offset");

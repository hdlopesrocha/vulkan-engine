#pragma once
#include <glm/glm.hpp>

// GPU-side water params UBO (matches shader WaterParamsUBO layout)
struct WaterParamsGPU {
    glm::vec4 params1;  // x=refractionStrength, y=fresnelPower, z=transparency, w=reflectionStrength
    glm::vec4 params2;  // x=waterTint, y=noiseScale, z=noiseOctaves, w=noisePersistence
    glm::vec4 params3;  // x=noiseTimeSpeed, y=noiseLacunarity, z=specularIntensity, w=specularPower
    glm::vec4 shallowColor; // xyz = shallowColor, w = waveDepthTransition
    glm::vec4 deepColor; // xyz = deepColor, w = glitterIntensity
    glm::vec4 waveParams; // x=tessNoiseInfluence, y=unused, z=bumpAmplitude, w=depthFalloff
    glm::vec4 reserved1;  // x=enableReflection, y=enableRefraction, z=legacy enableBlur (unused), w=legacy blurRadius (unused)
    glm::vec4 reserved2;  // x=legacy blurSamples (unused), y=legacy volumeBlurRate (unused), z=volumeBumpRate, w=unused
    glm::vec4 reserved3;  // unused (legacy cubemap-available flag removed with Solid360)
    glm::vec4 tessParams; // x=tessNearDist, y=tessFarDist, z=tessMinLevel, w=tessMaxLevel
    glm::vec4 causticColor; // xyz = color of the focused sunlight, w = unused
    glm::vec4 causticParams; // x=softness (|J| floor), y=intensity, z=unused, w=tint depth reference
    glm::vec4 causticExtraParams; // reserved (wave-shape caustics need no mode/line/speed knobs)
    glm::vec4 absorptionParams; // xyz = Beer-Lambert coeff, w = absorption scale
    glm::vec4 refractionParams; // x=IOR, y=max thickness cap, z=shore fade depth, w=unused

    // ── Shore-wave system (all zones/thresholds configurable, no shader
    //    magic numbers; mirrors the GLSL WaterParamsGPU struct). ──
    glm::vec4 waveToggles;      // x=enableWaves, y=enableFoam, z=enableVolumetric, w=unused
    glm::vec4 waveZones;        // x=deep depth(>=), y=break depth, z=shallow depth, w=unused
    glm::vec4 waveDirection;    // xy=shore direction (unit, world XZ), zw=unused
    glm::vec4 waveShape;        // x=sharp deep, y=sharp break, z=sharp shallow, w=shoal gain
    glm::vec4 waveShoal;        // x=speed shoal, y=shallow decay, z=line amplitude, w=breaker width
    glm::vec4 waveComponent1;   // x=frequency(rad/m), y=speed(m/s), z=amplitude, w=unused
    glm::vec4 waveComponent2;   // x=cross frequency, y=cross speed, z=cross amp, w=cross phase offset along shore (world units)
    glm::vec4 waveBreaker;      // x=breaker amplitude, y=chop amount, z=whitecap onset, w=unused
    glm::vec4 waveCurl;         // x=breaker lip skew, y=breaker crest-line hook, zw=unused
    glm::vec4 waveWarp;         // x=crest phase warp(feature units), y=crest amp variation, z=ridge stretch(along/across), w=shore gradient step(texels)
    glm::vec4 waveMask;         // x=scale, y=threshold, z=softness, w=time speed
    glm::vec4 foamParams;       // x=crest threshold, y=trail phase, z=decay/m, w=color amount
    glm::vec4 foamNoise;        // x=scale, y=time speed, z=noise amount, w=shore amount
    glm::vec4 foamExtra;        // x=mask floor, y=diffuse floor, z=ambient, w=unused
    glm::vec4 foamContact;      // x=contact width(world), y=contact amount, z=contact alpha, w=unused
    glm::vec4 foamColor;        // rgb=foam color, a=unused
    glm::vec4 oceanColor;       // rgb=deep ocean color, a=ocean tint start depth
    glm::vec4 oceanParams;      // x=ocean depth scale, yzw=unused
    glm::vec4 volumetricParams; // x=strength, y=density, z=Henyey-Greenstein g, w=unused
    glm::vec4 volumetricColor;  // rgb=volumetric scatter tint, a=unused
};

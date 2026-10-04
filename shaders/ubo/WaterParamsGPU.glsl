#ifndef WATER_PARAMS_G_P_U_GLSL
#define WATER_PARAMS_G_P_U_GLSL

// Extracted from shaders/includes/ubo.glsl (single-struct GLSL type).

struct WaterParamsGPU {
    vec4 params1;  // x=refractionStrength, y=fresnelPower, z=transparency, w=reflectionStrength
    vec4 params2;  // x=waterTint, y=noiseScale, z=noiseOctaves, w=noisePersistence
    vec4 params3;  // x=noiseTimeSpeed, y=noiseLacunarity, z=specularIntensity, w=specularPower
    vec4 glitterParams; // x=glitterIntensity, yzw=unused
    vec4 blurParams; // x=enableBlur, y=max radius (pixels), z=radius per meter (px/m), w=unused
    vec4 waveParams; // x=tessNoiseInfluence, y=unused, z=waveAmplitude, w=depthFalloff
    vec4 reserved1;  // x=enableReflection, y=enableRefraction, zw=unused
    vec4 reserved2;  // w=uniformReflection, xyz=unused
    vec4 reserved3;  // x=cube360Available, yzw=unused
    vec4 tessParams; // x=tessNearDist, y=tessFarDist, z=tessMinLevel, w=tessMaxLevel
    vec4 causticColor; // rgb = caustic tint, w = unused
    vec4 causticParams; // x = softness (|J| floor), y = intensity, zw = unused
    vec4 causticExtraParams; // reserved (wave-shape caustics: no mode/line/speed knobs)
    vec4 absorptionParams; // xyz = Beer-Lambert coeff, w = absorption scale
    vec4 refractionParams; // x = IOR, y = max thickness cap, z = shore fade depth, w = unused

    // Shore-wave system (mirrors vulkan/ubo/WaterParamsGPU.hpp).
    vec4 waveToggles;      // x=enableWaves, y=enableFoam, z=enableVolumetric, w=unused
    vec4 waveZones;        // x=deep depth(>=), y=break depth, z=shallow depth, w=unused
    vec4 waveDirection;    // xy=shore direction (unit, world XZ), zw=unused
    vec4 waveShape;        // reserved (was: zone crest sharpness / shoal gain)
    vec4 waveShoal;        // xyz reserved, w=breaker tint band half-width
    vec4 waveComponent1;   // x=crest scale(1/period), y=speed(m/s), z=amplitude, w=unused
    vec4 waveComponent2;   // reserved (was: the second/cross swell train)
    vec4 waveBreaker;      // reserved (was: breaker bump / chop / whitecap / height falloff)
    vec4 waveCurl;         // reserved (was: breaker lip skew / crest hook)
    vec4 waveWarp;         // xyz reserved, w=shore gradient step(texels)
    vec4 waveMask;         // x=scale, y=threshold, z=softness, w=time speed
    vec4 foamParams;       // x=crest threshold, y=trail phase, z=decay/m, w=color amount
    vec4 foamNoise;        // x=scale, y=time speed, z=noise amount, w=shore amount
    vec4 foamExtra;        // x=mask floor, y=diffuse floor, z=ambient, w=unused
    vec4 foamContact;      // x=contact width(world), y=contact amount, z=contact alpha, w=contact pulse floor
    vec4 foamShape;        // x=edge hardness, y=coverage, zw=reserved
    vec4 foamColor;        // rgb=foam color, a=unused
    vec4 volumetricParams; // x=strength, y=density, z=Henyey-Greenstein g, w=unused
    vec4 volumetricColor;  // rgb=volumetric scatter tint, a=unused

    // Depth-region tint (mirrors vulkan/ubo/WaterParamsGPU.hpp)   // rgb = tint at the waterline (d < zoneShallow) // rgb = foam-decay-band tint (zoneShallow..zoneBreak) // rgb = breaker-line tint (around zoneBreak)   // rgb = shoaling-band tint (zoneBreak..zoneDeep)    // rgb = open-ocean tint (d >= zoneDeep)
    vec4 regionShoreColor;   // rgb = the single water colour (one region)
    vec4 regionShallowColor; // reserved
    vec4 regionBreakerColor; // reserved
    vec4 regionShoalColor;   // reserved
    vec4 regionDeepColor;    // reserved
    vec4 regionTintParams;   // x=unused, y=tint shore fade depth (m), zw=unused   // x=blend softness, y=tint shore fade depth (m), zw=unused

    // Music-reactive audio input (mirrors vulkan/ubo/WaterParamsGPU.hpp):
    // written per frame by the MusicWidget analysis pipeline.
    vec4 musicAudio1; // x=smoothed audioAmplitude 0..1, y=bassEnergy, z=midEnergy, w=highEnergy
    vec4 musicAudio2; // x=beatIntensity 0..1, y=reactive input enabled 1/0, zw=reserved
};

#endif // WATER_PARAMS_G_P_U_GLSL

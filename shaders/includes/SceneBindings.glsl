
#include "../ssbo/Material.glsl"
#include "../types/MaterialNamed.glsl"
#include "../ubo/SkyUniform.glsl"
#include "../ubo/UniformObject.glsl"
#include "../ssbo/WaterParams.glsl"
#include "../types/WaterParamsNamed.glsl"
#include "../ubo/WaterRenderUBO.glsl"
// UBO layout must match the CPU-side UniformObject (std140-like):
// mat4 viewProjection; vec4 viewPos; vec4 lightDir; vec4 lightColor;
// Canonical scene UBO (set=0 binding=0): the shared struct IS the block
// layout — no packed/named-view split.
layout(std140, set = 0, binding = 0) uniform SolidParamsBlock {
    UniformObject ubo;
};


layout(std430, set = 0, binding = 5) readonly buffer Materials {
    Material materials[];
};


MaterialNamed materialNamed(Material m) {
    MaterialNamed n;
    n.skipEnvMap = m.materialFlags.x > 0.5;
    n.ambientFactor = m.materialFlags.z;
    n.mappingEnabled = m.mappingParams.x > 0.5;
    n.tessLevel = m.mappingParams.y;
    n.invertHeight = m.mappingParams.z > 0.5;
    n.tessHeightScale = m.mappingParams.w;
    n.specularStrength = m.specularParams.x;
    n.shininess = m.specularParams.y;
    n.triplanarScaleU = m.triplanarParams.x;
    n.triplanarScaleV = m.triplanarParams.y;
    n.triplanarEnabled = m.triplanarParams.z > 0.5;
    n.flipNormalY = m.normalParams.x > 0.5;
    n.swapNormalXZ = m.normalParams.y > 0.5;
    n.invertWidth = m.normalParams.z > 0.5;
    n.minLevel = m.tessLevelParams.x;
    n.maxLevel = m.tessLevelParams.y;
    n.reflectionStrength = m.tessLevelParams.z;
    n.roughnessFactor = m.roughnessAOParams.x;
    n.aoFactor = m.roughnessAOParams.y;
    n.useAO = m.roughnessAOParams.z > 0.5;
    return n;
}

// Per-draw model matrices for indirect rendering
// Models SSBO removed — shaders use identity models

// Dedicated UBO for skysphere parameters. Bound separately so sky shaders
// can read a small, focused uniform block instead of the large scene UBO.
// The cloud block extends the same UBO (no new descriptor binding).
// Canonical sky/cloud block (set=0 binding=6): the shared struct IS the
// block layout — no packed/named-view split.
layout(std140, set = 0, binding = 6) uniform SkyUBOBlock {
    SkyUniform sky;
};

// Canonical water render block (set=0 binding=10): the shared struct IS the
// block layout — no packed/named-view split.
layout(std140, set = 0, binding = 10) uniform WaterRenderBlock {
    WaterRenderUBO waterRenderUBO;
};



layout(std430, set = 0, binding = 7) readonly buffer WaterParamsBlock {
    WaterParams waterParams[];
};

WaterParamsNamed waterParamsNamed(WaterParams p) {
    WaterParamsNamed n;
    n.enableReflection = p.reserved1.x > 0.5;
    n.enableRefraction = p.reserved1.y > 0.5;
    n.uniformReflection = p.reserved2.w > 0.5;
    n.enableBlur = p.blurParams.x > 0.5;
    n.enableWaves = p.waveToggles.x > 0.5;
    n.enableFoam = p.waveToggles.y > 0.5;
    n.enableVolumetric = p.waveToggles.z > 0.5;
    n.refractionStrength = p.params1.x;
    n.fresnelPower = p.params1.y;
    n.transparency = p.params1.z;
    n.reflectionStrength = p.params1.w;
    n.waterTint = p.params2.x;
    n.noiseScale = p.params2.y;
    n.noiseOctaves = p.params2.z;
    n.noisePersistence = p.params2.w;
    n.noiseTimeSpeed = p.params3.x;
    n.noiseLacunarity = p.params3.y;
    n.specularIntensity = p.params3.z;
    n.specularPower = p.params3.w;
    n.glitterIntensity = p.glitterParams.x;
    n.blurRadius = p.blurParams.y;
    n.blurDepthScale = p.blurParams.z;
    n.tessNoiseInfluence = p.waveParams.x;
    n.bumpAmplitude = p.waveParams.z;
    n.depthFalloff = p.waveParams.w;
    n.tessNearDist = p.tessParams.x;
    n.tessFarDist = p.tessParams.y;
    n.tessMinLevel = p.tessParams.z;
    n.tessMaxLevel = p.tessParams.w;
    n.causticColor = p.causticColor.rgb;
    n.causticSoftness = p.causticParams.x;
    n.causticIntensity = p.causticParams.y;
    n.absorption = p.absorptionParams.xyz;
    n.absorptionScale = p.absorptionParams.w;
    n.waterIor = p.refractionParams.x;
    n.maxThickness = p.refractionParams.y;
    n.shoreFadeDepth = p.refractionParams.z;
    n.waveDirection = p.waveDirection.xy;
    n.breakerWidth = p.waveShoal.w;
    n.wavePeriodScale = p.waveComponent1.x;
    n.shoreWaveFade = p.waveMask.x;
    n.shoreWaveSlope = p.waveShape.x;
    n.waveSteepness = p.waveShoal.z;
    n.waveCrestSharpness = p.waveShape.y;
    n.rippleHeight = p.waveCurl.x;
    n.waterColor = p.regionShoreColor.rgb;
    n.waveSpeed = p.waveComponent1.y;
    n.waveAmplitude = p.waveComponent1.z;
    n.shoreGradientStep = p.waveWarp.w;
    n.foamCrestThreshold = p.foamParams.x;
    n.foamTrailPhase = p.foamParams.y;
    n.foamDecay = p.foamParams.z;
    n.foamColorAmount = p.foamParams.w;
    n.foamNoisePeriodScale = p.foamNoise.x;
    n.foamNoiseSpeed = p.foamNoise.y;
    n.foamNoiseAmount = p.foamNoise.z;
    n.foamShoreAmount = p.foamNoise.w;
    n.foamMaskFloor = p.foamExtra.x;
    n.foamDiffuseFloor = p.foamExtra.y;
    n.foamAmbient = p.foamExtra.z;
    n.foamContactWidth = p.foamContact.x;
    n.foamContactAmount = p.foamContact.y;
    n.foamContactAlpha = p.foamContact.z;
    n.foamContactFloor = p.foamContact.w;
    n.foamEdge = p.foamShape.x;
    n.foamCoverage = p.foamShape.y;
    n.foamColor = p.foamColor.rgb;
    n.volumetricStrength = p.volumetricParams.x;
    n.volumetricDensity = p.volumetricParams.y;
    n.volumetricPhaseG = p.volumetricParams.z;
    n.volumetricColor = p.volumetricColor.rgb;
    n.tintShoreFadeDepth = p.regionTintParams.y;
    n.audioAmplitude = p.musicAudio1.x;
    n.bassEnergy = p.musicAudio1.y;
    n.midEnergy = p.musicAudio1.z;
    n.highEnergy = p.musicAudio1.w;
    n.beatIntensity = p.musicAudio2.x;
    n.musicReactive = p.musicAudio2.y > 0.5;
    return n;
}

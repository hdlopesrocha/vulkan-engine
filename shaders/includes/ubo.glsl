
#include "../ubo/MaterialGPU.glsl"
#include "../ubo/MaterialNamed.glsl"
#include "../ubo/SkyParamsNamed.glsl"
#include "../ubo/UniformObjectNamed.glsl"
#include "../ubo/WaterParamsGPU.glsl"
#include "../ubo/WaterParamsNamed.glsl"
#include "../ubo/WaterRenderParamsNamed.glsl"
// UBO layout must match the CPU-side UniformObject (std140-like):
// mat4 viewProjection; vec4 viewPos; vec4 lightDir; vec4 lightColor;
layout(set = 0, binding = 0) uniform SolidParamsUBO {
    mat4 viewProjection;
    vec4 viewPos;
    vec4 lightDir;
    vec4 lightColor;
    vec4 materialFlags;
    mat4 lightSpaceMatrix; // for shadow mapping
    vec4 shadowEffects; // x/y/z = unused, w=global shadows enabled (1.0 = on)
    vec4 debugParams; // x=DebugMode (see includes/debug_modes.glsl; 0=default render); y=roughnessEnabled, z=aoEnabled
    vec4 triplanarSettings;
    vec4 tessParams; // x = tessNearDist, y = tessFarDist, z = tessellationFactor, w = reserved
    vec4 passParams;   // x = isShadowPass, y = tessEnabled, z = nearPlane, w = farPlane
    mat4 lightSpaceMatrix1; // cascade 1 (4x ortho0)
    mat4 lightSpaceMatrix2; // cascade 2 (16x ortho0)
    mat4 invViewProjection; // inverse of viewProjection (camera-constant)
    vec4 brushParams;       // x=brushTextureIndex, y=brushMode (0=overlay, 2=PAINT)
    vec4 brushHSV;          // x=H(0..360), y=S(0..1), z=V(0..1), w=unused
} uboPacked;


UniformObjectNamed uniformObjectNamed() {
    UniformObjectNamed n;
    n.viewProjection = uboPacked.viewProjection;
    n.viewPosition = uboPacked.viewPos.xyz;
    n.lightDirection = uboPacked.lightDir.xyz;
    n.lightElevation = uboPacked.lightDir.y;
    n.lightColor = uboPacked.lightColor.xyz;
    n.cubemapCapture = uboPacked.materialFlags.x > 0.5;
    n.normalMappingEnabled = uboPacked.materialFlags.w > 0.5;
    n.shadowsEnabled = uboPacked.shadowEffects.w > 0.5;
    n.debugMode = int(uboPacked.debugParams.x + 0.5);
    n.roughnessEnabled = uboPacked.debugParams.y > 0.5;
    n.ambientOcclusionEnabled = uboPacked.debugParams.z > 0.5;
    n.triplanarThreshold = uboPacked.triplanarSettings.x;
    n.triplanarExponent = uboPacked.triplanarSettings.y;
    n.tessNearDist = uboPacked.tessParams.x;
    n.tessFarDist = uboPacked.tessParams.y;
    n.tessellationFactor = uboPacked.tessParams.z;
    n.isShadowPass = uboPacked.passParams.x > 0.5;
    n.tessellationEnabled = uboPacked.passParams.y > 0.5;
    n.nearPlane = uboPacked.passParams.z;
    n.farPlane = uboPacked.passParams.w;
    n.lightSpaceMatrix = uboPacked.lightSpaceMatrix;
    n.lightSpaceMatrix1 = uboPacked.lightSpaceMatrix1;
    n.lightSpaceMatrix2 = uboPacked.lightSpaceMatrix2;
    n.invViewProjection = uboPacked.invViewProjection;
    n.brushTextureIndex = uboPacked.brushParams.x;
    n.brushMode = uboPacked.brushParams.y;
    n.brushPhase = uboPacked.brushParams.w;
    n.brushHsv = uboPacked.brushHSV.xyz;
    return n;
}

UniformObjectNamed ubo = uniformObjectNamed();


layout(std430, set = 0, binding = 5) readonly buffer Materials {
    MaterialGPU materials[];
};


MaterialNamed materialNamed(MaterialGPU m) {
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
layout(set = 0, binding = 6) uniform SkyUBO {
    vec4 skyHorizon; // rgb = horizon color, a = unused
    vec4 skyZenith;  // rgb = zenith color, a = unused
    vec4 skyParams;  // x = warmth, y = exponent, z = sunFlare, w = skyMode (0=gradient, 1=grid)
    vec4 nightHorizon; // rgb = night horizon color
    vec4 nightZenith;  // rgb = night zenith color
    vec4 nightParams;  // x = night intensity (0..1), y = starIntensity, z/w unused
    vec4 cloudToggles; // x = enabled, y = lowOn, z = midOn, w = highOn
    vec4 cloudGlobal;  // x = densityScale, y = windSpeed, z = windAngleRad, w = detailStrength
    vec4 cloudTime;    // x = time, y = shadowStrength, z = raymarchSteps, w = lightSteps
    vec4 cloudLow;     // x = coverage, y = density, z = scale, w = windSpeedMul
    vec4 cloudLowGeom; // x = baseHeight, y = thickness, zw unused
    vec4 cloudMid;     // x = coverage, y = density, z = scale, w = windSpeedMul
    vec4 cloudMidGeom; // x = baseHeight, y = thickness, zw unused
    vec4 cloudHigh;    // x = coverage, y = density, z = scale, w = windSpeedMul
    vec4 cloudHighGeom;// x = baseHeight, y = thickness, zw unused
    vec4 cloudLight;   // x = silverLining, y = ambientBoost, z = sunForwardG, w = exposure
    vec4 cloudAnim;    // x = timeScale, yzw unused
} skyPacked;


SkyParamsNamed skyParamsNamed() {
    SkyParamsNamed n;
    n.horizonColor = skyPacked.skyHorizon.rgb;
    n.zenithColor = skyPacked.skyZenith.rgb;
    n.warmth = skyPacked.skyParams.x;
    n.exponent = skyPacked.skyParams.y;
    n.sunFlare = skyPacked.skyParams.z;
    n.nightHorizonColor = skyPacked.nightHorizon.rgb;
    n.nightZenithColor = skyPacked.nightZenith.rgb;
    n.nightIntensity = skyPacked.nightParams.x;
    n.starIntensity = skyPacked.nightParams.y;
    n.cloudsEnabled = skyPacked.cloudToggles.x > 0.5;
    n.lowEnabled = skyPacked.cloudToggles.y > 0.5;
    n.midEnabled = skyPacked.cloudToggles.z > 0.5;
    n.highEnabled = skyPacked.cloudToggles.w > 0.5;
    n.densityScale = skyPacked.cloudGlobal.x;
    n.windSpeed = skyPacked.cloudGlobal.y;
    n.windAngleRad = skyPacked.cloudGlobal.z;
    n.detailStrength = skyPacked.cloudGlobal.w;
    n.cloudTime = skyPacked.cloudTime.x;
    n.shadowStrength = skyPacked.cloudTime.y;
    n.raymarchSteps = skyPacked.cloudTime.z;
    n.lightSteps = skyPacked.cloudTime.w;
    n.lowTier = skyPacked.cloudLow;
    n.lowGeom = skyPacked.cloudLowGeom.xy;
    n.midTier = skyPacked.cloudMid;
    n.midGeom = skyPacked.cloudMidGeom.xy;
    n.highTier = skyPacked.cloudHigh;
    n.highGeom = skyPacked.cloudHighGeom.xy;
    n.silverLining = skyPacked.cloudLight.x;
    n.ambientBoost = skyPacked.cloudLight.y;
    n.sunForwardG = skyPacked.cloudLight.z;
    n.exposure = skyPacked.cloudLight.w;
    return n;
}

layout(set = 0, binding = 10) uniform WaterRenderUBO {
    vec4 timeParams; // x=waterTime, y=water refraction allowed, z=water reflection allowed, w=water blur allowed
    vec4 depthParams; // x = solidSceneDepthTex is THIS frame's solid depth (1/0); yzw unused
} waterRenderUBOPacked;


WaterRenderParamsNamed waterRenderParamsNamed() {
    WaterRenderParamsNamed n;
    n.waterTime = waterRenderUBOPacked.timeParams.x;
    n.refractionAllowed = waterRenderUBOPacked.timeParams.y > 0.5;
    n.reflectionAllowed = waterRenderUBOPacked.timeParams.z > 0.5;
    n.blurAllowed = waterRenderUBOPacked.timeParams.w > 0.5;
    n.solidDepthIsCurrent = waterRenderUBOPacked.depthParams.x > 0.5;
    return n;
}



layout(std430, set = 0, binding = 7) readonly buffer WaterParamsBlock {
    WaterParamsGPU waterParams[];
};

WaterParamsNamed waterParamsNamed(WaterParamsGPU p) {
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

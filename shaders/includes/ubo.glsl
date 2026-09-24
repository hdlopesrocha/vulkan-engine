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

// Named view over the packed SolidParamsUBO - same data, descriptive names. The builder below is the
// only place the packed component letters are read; every other access uses the
// named attributes.
struct UniformObjectNamed {
    mat4 viewProjection;
    vec3 viewPosition;
    vec3 lightDirection;
    float lightElevation;
    vec3 lightColor;
    bool cubemapCapture;
    bool normalMappingEnabled;
    bool shadowsEnabled;
    int debugMode;
    bool roughnessEnabled;
    bool ambientOcclusionEnabled;
    float triplanarThreshold;
    float triplanarExponent;
    float tessNearDist;
    float tessFarDist;
    float tessellationFactor;
    bool isShadowPass;
    bool tessellationEnabled;
    float nearPlane;
    float farPlane;
    mat4 lightSpaceMatrix;
    mat4 lightSpaceMatrix1;
    mat4 lightSpaceMatrix2;
    mat4 invViewProjection;
    float brushTextureIndex;
    float brushMode;
    float brushPhase;
    vec3 brushHsv;
};

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

// Packed material data uploaded once to GPU. Matches the CPU-side MaterialGPU (6 vec4s).
// Access this as `materials[brushIndex]` from shaders. Uses std430 for tightly-packed vec4 alignment.
struct MaterialGPU {
    vec4 materialFlags;    // .x = skipEnvMap (set during cubemap capture), .z = ambientFactor
    vec4 mappingParams;    // x = mappingEnabled (0/1), y = tessLevel, z = invertHeight (0/1), w = tessHeightScale
    vec4 specularParams;   // x = specularStrength, y = shininess
    vec4 triplanarParams;  // x = scaleU, y = scaleV, z = triplanarEnabled (0/1)
    vec4 normalParams;     // x = flipNormalY (0/1), y = swapNormalXZ (0/1), z = invertWidth (0/1)
    vec4 tessLevelParams;  // x = minLevel, y = maxLevel, z = reflectionStrength, w = reserved
    vec4 roughnessAOParams; // x = roughnessFactor, y = aoFactor, z = useAO (1.0/0.0)
};

layout(std430, set = 0, binding = 5) readonly buffer Materials {
    MaterialGPU materials[];
};

// Named view over the packed MaterialGPU - same data, descriptive names. The builder below is the
// only place the packed component letters are read; every other access uses the
// named attributes.
struct MaterialNamed {
    bool skipEnvMap;
    float ambientFactor;
    bool mappingEnabled;
    float tessLevel;
    bool invertHeight;
    float tessHeightScale;
    float specularStrength;
    float shininess;
    float triplanarScaleU;
    float triplanarScaleV;
    bool triplanarEnabled;
    bool flipNormalY;
    bool swapNormalXZ;
    bool invertWidth;
    float minLevel;
    float maxLevel;
    float reflectionStrength;
    float roughnessFactor;
    float aoFactor;
    bool useAO;
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
layout(set = 0, binding = 6) uniform SkyUBO {
    vec4 skyHorizon; // rgb = horizon color, a = unused
    vec4 skyZenith;  // rgb = zenith color, a = unused
    vec4 skyParams;  // x = warmth, y = exponent, z = sunFlare, w = skyMode (0=gradient, 1=grid)
    vec4 nightHorizon; // rgb = night horizon color
    vec4 nightZenith;  // rgb = night zenith color
    vec4 nightParams;  // x = night intensity (0..1), y = starIntensity, z/w unused
} skyPacked;

// Named view over the packed SkyUBO - same data, descriptive names. The builder below is the
// only place the packed component letters are read; every other access uses the
// named attributes.
struct SkyParamsNamed {
    vec3 horizonColor;
    vec3 zenithColor;
    float warmth;
    float exponent;
    float sunFlare;
    vec3 nightHorizonColor;
    vec3 nightZenithColor;
    float nightIntensity;
    float starIntensity;
};

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
    return n;
}

layout(set = 0, binding = 10) uniform WaterRenderUBO {
    vec4 timeParams; // x=waterTime, y=water refraction allowed, z=water reflection allowed, w=water blur allowed
    vec4 depthParams; // x = solidSceneDepthTex is THIS frame's solid depth (1/0); yzw unused
} waterRenderUBOPacked;

// Named view over the packed WaterRenderUBO - same data, descriptive names. The builder below is the
// only place the packed component letters are read; every other access uses the
// named attributes.
struct WaterRenderParamsNamed {
    float waterTime;
    bool refractionAllowed;
    bool reflectionAllowed;
    bool blurAllowed;
    bool solidDepthIsCurrent;
};

WaterRenderParamsNamed waterRenderParamsNamed() {
    WaterRenderParamsNamed n;
    n.waterTime = waterRenderUBOPacked.timeParams.x;
    n.refractionAllowed = waterRenderUBOPacked.timeParams.y > 0.5;
    n.reflectionAllowed = waterRenderUBOPacked.timeParams.z > 0.5;
    n.blurAllowed = waterRenderUBOPacked.timeParams.w > 0.5;
    n.solidDepthIsCurrent = waterRenderUBOPacked.depthParams.x > 0.5;
    return n;
}


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
};

layout(std430, set = 0, binding = 7) readonly buffer WaterParamsBlock {
    WaterParamsGPU waterParams[];
};
// ── Named view over the packed WaterParamsGPU. ───────────────────────────
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
    return n;
}

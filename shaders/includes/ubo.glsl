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
} ubo;

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
} sky;

layout(set = 0, binding = 10) uniform WaterRenderUBO {
    vec4 timeParams; // x=waterTime, y=water refraction allowed, z=water reflection allowed, w=water blur allowed
} waterRenderUBO;


struct WaterParamsGPU {
    vec4 params1;  // x=refractionStrength, y=fresnelPower, z=transparency, w=reflectionStrength
    vec4 params2;  // x=waterTint, y=noiseScale, z=noiseOctaves, w=noisePersistence
    vec4 params3;  // x=noiseTimeSpeed, y=noiseLacunarity, z=specularIntensity, w=specularPower
    vec4 glitterParams; // x=glitterIntensity, yzw=unused
    vec4 blurParams; // x=enableBlur, y=max radius (pixels), z=radius per meter (px/m), w=unused
    vec4 waveParams; // x=tessNoiseInfluence, y=unused, z=bumpAmplitude, w=depthFalloff
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
    vec4 waveShape;        // x=sharp deep, y=sharp break, z=sharp shallow, w=shoal gain
    vec4 waveShoal;        // x=speed shoal, y=shallow decay, z=line amplitude, w=breaker width
    vec4 waveComponent1;   // x=crest scale(1/period), y=speed(m/s), z=amplitude, w=unused
    vec4 waveComponent2;   // x=cross crest scale(1/period), y=cross speed, z=cross amp, w=cross phase offset along shore (world units)
    vec4 waveBreaker;      // x=breaker amplitude, y=chop amount, z=whitecap onset, w=height falloff exponent
    vec4 waveCurl;         // x=breaker lip skew, y=breaker crest-line hook, zw=unused
    vec4 waveWarp;         // x=crest phase warp(feature units), y=crest amp variation, z=ridge stretch(along/across), w=shore gradient step(texels)
    vec4 waveMask;         // x=scale, y=threshold, z=softness, w=time speed
    vec4 foamParams;       // x=crest threshold, y=trail phase, z=decay/m, w=color amount
    vec4 foamNoise;        // x=scale, y=time speed, z=noise amount, w=shore amount
    vec4 foamExtra;        // x=mask floor, y=diffuse floor, z=ambient, w=unused
    vec4 foamContact;      // x=contact width(world), y=contact amount, z=contact alpha, w=contact pulse floor
    vec4 foamShape;        // x=edge hardness, y=coverage, z=shore speed factor, w=trail lag growth
    vec4 foamColor;        // rgb=foam color, a=unused
    vec4 volumetricParams; // x=strength, y=density, z=Henyey-Greenstein g, w=unused
    vec4 volumetricColor;  // rgb=volumetric scatter tint, a=unused

    // Depth-region tint (mirrors vulkan/ubo/WaterParamsGPU.hpp)
    vec4 regionShoreColor;   // rgb = tint at the waterline (d < zoneShallow)
    vec4 regionShallowColor; // rgb = foam-decay-band tint (zoneShallow..zoneBreak)
    vec4 regionBreakerColor; // rgb = breaker-line tint (around zoneBreak)
    vec4 regionShoalColor;   // rgb = shoaling-band tint (zoneBreak..zoneDeep)
    vec4 regionDeepColor;    // rgb = open-ocean tint (d >= zoneDeep)
    vec4 regionTintParams;   // x=blend softness, y=tint shore fade depth (m), zw=unused
};

layout(std430, set = 0, binding = 7) readonly buffer WaterParamsBlock {
    WaterParamsGPU waterParams[];
};
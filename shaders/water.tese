#version 450

// Water tessellation evaluation shader
// Applies wave displacement using Perlin noise

layout(triangles, equal_spacing, cw) in;

layout(location = 0) in vec3 inPos[];
layout(location = 1) in vec3 inNormal[];
layout(location = 2) in vec2 inTexCoord[];

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;
layout(location = 3) out vec4 fragPosClip;  // clip-space position for depth lookup
layout(location = 4) out vec3 fragDebug;   // debug visual (displacement)

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 viewProjection;
    vec4 viewPos;
    vec4 lightDir;
    vec4 lightColor;
    vec4 materialFlags;
    mat4 lightSpaceMatrix;
    vec4 shadowEffects;
    vec4 debugParams;
    vec4 triplanarSettings;
    vec4 tessParams;   // x=nearDist, y=farDist, z=minLevel, w=maxLevel
    vec4 passParams;   // x=isShadowPass, y=tessEnabled, z=nearPlane, w=farPlane
} ubo;

// Water-specific parameters (set 0, binding = 7) - same layout as the post-process shader
layout(set = 0, binding = 7) uniform WaterParamsUBO {
    vec4 params1;  // x=refractionStrength, y=fresnelPower, z=transparency, w=foamDepthThreshold
    vec4 params2;  // x=waterTint, y=noiseScale, z=noiseOctaves, w=noisePersistence
    vec4 params3;  // x=noiseTimeSpeed, y=waterTime, z=shoreStrength, w=shoreFalloff
    vec4 shallowColor; // xyz = shallowColor, w = waveDepthTransition
    vec4 deepColor; // xyz = deepColor, w = unused
    vec4 waveParams; // x=waveNoiseScale, y=waveNoiseOctaves, z=waveNoisePersistence, w=unused
    vec4 waveParams2; // x=unused, y=unused, z=bumpAmplitude, w=depthFalloff
    vec4 reserved;   // unused
} waterParams;

// Scene depth texture for depth-dependent wave attenuation (set 2)
layout(set = 2, binding = 1) uniform sampler2D sceneDepthTex;

#include "includes/perlin.glsl"
#include "includes/water_noise.glsl"

// Linearize depth from Vulkan [0,1] depth buffer to eye-space distance.
float linearizeDepth(float depth) {
    float nearPlane = ubo.passParams.z;
    float farPlane  = ubo.passParams.w;
    return (nearPlane * farPlane) / (farPlane - depth * (farPlane - nearPlane));
}

vec3 clampAndNormalizeBary(vec3 b) {
    b = max(b, vec3(0.0001));
    return b / (b.x + b.y + b.z);
}

vec3 sampleDisplacedPos(vec3 bary, float animTime,
                        float foamNoiseScale, int foamNoiseOctaves, float foamNoisePersistence,
                        float bumpAmp, float waveScale) {
    vec3 p = bary.x * inPos[0] + bary.y * inPos[1] + bary.z * inPos[2];
    vec3 n = normalize(bary.x * inNormal[0] + bary.y * inNormal[1] + bary.z * inNormal[2]);
    float h = waterWaveDisplacement(p.xyz, animTime, foamNoiseScale, foamNoiseOctaves, foamNoisePersistence, bumpAmp, waveScale);
    return p + n * h;
}
void main() {
    // Interpolate position
        vec3 bary = gl_TessCoord;

        // Interpolate position
        vec3 pos = bary.x * inPos[0] +
                   bary.y * inPos[1] +
                   bary.z * inPos[2];
    
    // Interpolate normal
        vec3 normal = normalize(bary.x * inNormal[0] +
                                bary.y * inNormal[1] +
                                bary.z * inNormal[2]);
    
    // Interpolate texture coordinates
        fragTexCoord = bary.x * inTexCoord[0] +
                       bary.y * inTexCoord[1] +
                       bary.z * inTexCoord[2];
    
    // Get water and noise parameters
    float time = waterParams.params3.y;
    if (time == 0.0) time = ubo.passParams.x;
    // waveScale is no longer in passParams (z/w now carry nearPlane/farPlane).
    // Displacement magnitude is fully controlled by bumpAmp from the widget.
    float waveScale = 1.0;
    float noiseTimeSpeed = waterParams.params3.x;

    // foam/Noise params (shared with foam logic)
    float waveNoiseScale = 1.0 / max(waterParams.waveParams.x, 0.0001);
    int waveNoiseOctaves = int(max(waterParams.waveParams.y, 1.0));
    float waveNoisePersistence = waterParams.waveParams.z;

    float bumpAmp = waterParams.waveParams2.z; // bump amplitude provided via Water widget

    // --- Depth-based wave attenuation ---
    // Project the undisplaced water vertex to screen space and sample the solid
    // depth buffer.  Where the water surface is close to solid geometry (shallow),
    // waves are suppressed.  waveDepthTransition controls the ramp distance.
    float waveDepthTransition = waterParams.shallowColor.w;
    if (waveDepthTransition > 0.0) {
        vec4 preClip = ubo.viewProjection * vec4(pos, 1.0);
        if (preClip.w > 0.001) {
            vec2 screenUV = clamp(preClip.xy / preClip.w * 0.5 + 0.5, 0.001, 0.999);
            float solidDepthRaw = texture(sceneDepthTex, screenUV).r;
            float waterDepthRaw = preClip.z / preClip.w;
            float solidDepthLin = linearizeDepth(solidDepthRaw);
            float waterDepthLin = linearizeDepth(waterDepthRaw);
            float depthDiff = max(solidDepthLin - waterDepthLin, 0.0);
            bumpAmp *= smoothstep(0.0, waveDepthTransition, depthDiff);
        }
    }

    // Calculate wave displacement using 4D Perlin FBM (pos.xyz, time)
    float animTime = time * noiseTimeSpeed;
    vec3 xyz = pos.xyz;
        float waveDisplacement = waterWaveDisplacement(
            xyz,
            animTime,
            waveNoiseScale,
            waveNoiseOctaves,
            waveNoisePersistence,
            bumpAmp,
            waveScale
        );

    // Displace along the interpolated surface normal so bump follows mesh orientation.
    pos += waveDisplacement * normal;

    // Calculate perturbed normal from neighboring displaced points in barycentric space.
    // This is robust on curved meshes and avoids axis-aligned artifacts.
    float epsBary = 0.005;
    vec3 bUPlus  = clampAndNormalizeBary(bary + vec3( epsBary, -epsBary, 0.0));
    vec3 bUMinus = clampAndNormalizeBary(bary + vec3(-epsBary,  epsBary, 0.0));
    vec3 bVPlus  = clampAndNormalizeBary(bary + vec3(0.0,  epsBary, -epsBary));
    vec3 bVMinus = clampAndNormalizeBary(bary + vec3(0.0, -epsBary,  epsBary));

    vec3 pUPlus = sampleDisplacedPos(bUPlus, animTime, waveNoiseScale, waveNoiseOctaves, waveNoisePersistence, bumpAmp, waveScale);
    vec3 pUMinus = sampleDisplacedPos(bUMinus, animTime, waveNoiseScale, waveNoiseOctaves, waveNoisePersistence, bumpAmp, waveScale);
    vec3 pVPlus = sampleDisplacedPos(bVPlus, animTime, waveNoiseScale, waveNoiseOctaves, waveNoisePersistence, bumpAmp, waveScale);
    vec3 pVMinus = sampleDisplacedPos(bVMinus, animTime, waveNoiseScale, waveNoiseOctaves, waveNoisePersistence, bumpAmp, waveScale);

    vec3 tangentU = pUPlus - pUMinus;
    vec3 tangentV = pVPlus - pVMinus;

    fragNormal = normalize(cross(tangentU, tangentV));
    if (dot(fragNormal, normal) < 0.0) fragNormal = -fragNormal;

    // Debug: encode displacement as color (normalized)
    float maxExpected = bumpAmp * waveScale * 1.5; // heuristic normalization factor
    float normDisp = clamp((waveDisplacement / maxExpected) * 0.5 + 0.5, 0.0, 1.0);
    fragDebug = vec3(normDisp);
    
    fragPos = pos;
    vec4 clipPos = ubo.viewProjection * vec4(pos, 1.0);
    fragPosClip = clipPos;
    gl_Position = clipPos;
}

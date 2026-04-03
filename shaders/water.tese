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
    vec4 shallowColor;
    vec4 deepColor; // w = foamIntensity
    vec4 foamParams; // x=foamNoiseScale, y=foamNoiseOctaves, z=foamNoisePersistence, w=foamTintIntensity
    vec4 foamParams2; // x=foamBrightness, y=foamContrast, z=bumpAmplitude
    vec4 foamTint;   // rgb foam tint
} waterParams;

#include "includes/perlin.glsl"
#include "includes/water_noise.glsl"

vec3 clampAndNormalizeBary(vec3 b) {
    b = max(b, vec3(0.0001));
    return b / (b.x + b.y + b.z);
}

vec3 sampleDisplacedPos(vec3 bary, float animTime,
                        float foamNoiseScale, int foamNoiseOctaves, float foamNoisePersistence,
                        float bumpAmp, float waveScale) {
    vec3 p = bary.x * inPos[0] + bary.y * inPos[1] + bary.z * inPos[2];
    vec3 n = normalize(bary.x * inNormal[0] + bary.y * inNormal[1] + bary.z * inNormal[2]);
    float h = waterWaveDisplacement(p.xz, animTime, foamNoiseScale, foamNoiseOctaves, foamNoisePersistence, bumpAmp, waveScale);
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
    float foamNoiseScale = 1.0 / max(waterParams.foamParams.x, 0.0001);
    int foamNoiseOctaves = int(max(waterParams.foamParams.y, 1.0));
    float foamNoisePersistence = waterParams.foamParams.z;

    float bumpAmp = waterParams.foamParams2.z; // bump amplitude provided via Water widget

    // Calculate wave displacement using 4D Perlin FBM (pos.xz, time)
    float animTime = time * noiseTimeSpeed;
    vec2 xz = pos.xz;
        float waveDisplacement = waterWaveDisplacement(
            xz,
            animTime,
            foamNoiseScale,
            foamNoiseOctaves,
            foamNoisePersistence,
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

    vec3 pUPlus = sampleDisplacedPos(bUPlus, animTime, foamNoiseScale, foamNoiseOctaves, foamNoisePersistence, bumpAmp, waveScale);
    vec3 pUMinus = sampleDisplacedPos(bUMinus, animTime, foamNoiseScale, foamNoiseOctaves, foamNoisePersistence, bumpAmp, waveScale);
    vec3 pVPlus = sampleDisplacedPos(bVPlus, animTime, foamNoiseScale, foamNoiseOctaves, foamNoisePersistence, bumpAmp, waveScale);
    vec3 pVMinus = sampleDisplacedPos(bVMinus, animTime, foamNoiseScale, foamNoiseOctaves, foamNoisePersistence, bumpAmp, waveScale);

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

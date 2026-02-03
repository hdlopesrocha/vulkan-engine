#version 450

// Water tessellation evaluation shader
// Applies wave displacement using Perlin noise

layout(triangles, equal_spacing, ccw) in;

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
    vec4 passParams;   // x=time, y=tessEnabled, z=waveScale, w=noiseScale
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

void main() {
    // Interpolate position
    vec3 pos = gl_TessCoord.x * inPos[0] + 
               gl_TessCoord.y * inPos[1] + 
               gl_TessCoord.z * inPos[2];
    
    // Interpolate normal
    vec3 normal = normalize(gl_TessCoord.x * inNormal[0] + 
                           gl_TessCoord.y * inNormal[1] + 
                           gl_TessCoord.z * inNormal[2]);
    
    // Interpolate texture coordinates
    fragTexCoord = gl_TessCoord.x * inTexCoord[0] + 
                   gl_TessCoord.y * inTexCoord[1] + 
                   gl_TessCoord.z * inTexCoord[2];
    
    // Get water and noise parameters
    float time = waterParams.params3.y;
    if (time == 0.0) time = ubo.passParams.x;
    float waveScale = ubo.passParams.z;
    float noiseScale = ubo.passParams.w;
    float noiseTimeSpeed = waterParams.params3.x;

    // foam/Noise params (shared with foam logic)
    float foamNoiseScale = 1.0 / max(waterParams.foamParams.x, 0.0001);
    int foamNoiseOctaves = int(max(waterParams.foamParams.y, 1.0));
    float foamNoisePersistence = waterParams.foamParams.z;

    float bumpAmp = waterParams.foamParams2.z; // bump amplitude provided via Water widget

    // Calculate wave displacement using 4D Perlin FBM (pos.xz, time)
    float animTime = time * noiseTimeSpeed;
    float baseNoise = fbm(vec4(pos.xz * foamNoiseScale * 0.15, 0.0, animTime * 0.15), foamNoiseOctaves, foamNoisePersistence);
    float baseNoise2 = fbm(vec4((pos.xz + vec2(50.0)) * foamNoiseScale * 0.07, 0.0, animTime * 0.12), max(foamNoiseOctaves - 1, 1), foamNoisePersistence);

    float waveDisplacement = (baseNoise + baseNoise2 * 0.5) * bumpAmp * waveScale;

    // Displace position along normal (Y-up for water surface)
    pos.y += waveDisplacement;

    // Calculate perturbed normal from wave gradient using small offsets in X/Z
    float eps = 0.1;
    float nX = fbm(vec4(vec2(pos.x + eps, pos.z) * foamNoiseScale * 0.15, 0.0, animTime * 0.15), foamNoiseOctaves, foamNoisePersistence);
    float nZ = fbm(vec4(vec2(pos.x, pos.z + eps) * foamNoiseScale * 0.15, 0.0, animTime * 0.15), foamNoiseOctaves, foamNoisePersistence);

    float dX = (nX - baseNoise) / eps * bumpAmp * waveScale;
    float dZ = (nZ - baseNoise) / eps * bumpAmp * waveScale;

    // Perturbed normal
    fragNormal = normalize(vec3(-dX, 1.0, -dZ));

    // Debug: encode displacement as color (normalized)
    float maxExpected = bumpAmp * waveScale * 1.5; // heuristic normalization factor
    float normDisp = clamp((waveDisplacement / maxExpected) * 0.5 + 0.5, 0.0, 1.0);
    fragDebug = vec3(normDisp);
    
    fragPos = pos;
    vec4 clipPos = ubo.viewProjection * vec4(pos, 1.0);
    fragPosClip = clipPos;
    gl_Position = clipPos;
}

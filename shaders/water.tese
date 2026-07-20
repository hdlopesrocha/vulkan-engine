#version 450

#include "includes/locations.glsl"

// Water tessellation evaluation shader
// Applies wave displacement using Perlin noise

layout(triangles, equal_spacing, cw) in;

layout(location = VARY_LOCALPOS) in vec3 inPos[];
layout(location = VARY_NORMAL) in vec3 inNormal[];
layout(location = VARY_SHARPNORMAL) in vec3 inBaseNormal[];
layout(location = VARY_UV) in vec2 inTexCoord[];
layout(location = VARY_BRUSHPATCH) in ivec3 tc_fragBrushIndex[];
layout(location = VARY_TEXWEIGHTS) in vec3 tc_fragTexWeights[];

layout(location = VARY_LOCALPOS) out vec3 fragPos;
layout(location = VARY_NORMAL) out vec3 fragNormal;
layout(location = VARY_SHARPNORMAL) out vec3 fragBaseNormal;  // undisplaced base normal for per-fragment detail
layout(location = VARY_BASEPOS) out vec4 fragBasePos;        // xyz = undisplaced base position, w = final bump amplitude
layout(location = VARY_UV) out vec2 fragTexCoord;
layout(location = VARY_POSCLIP) out vec4 fragPosClip;  // clip-space position for depth lookup
layout(location = VARY_DEBUG) out vec3 fragDebug;   // debug visual (displacement)
layout(location = VARY_POSWORLD) out vec3 fragPosWorld;  // world-space position for shadow cascades
layout(location = VARY_POSLIGHT) out vec4 fragPosLightSpace; // light-space pos (cascade 0)
layout(location = VARY_BRUSHPATCH) flat out int fragBrushIndex;

#include "includes/ubo.glsl"

// Scene depth texture for depth-dependent wave attenuation (set 2)
layout(set = 2, binding = 1) uniform sampler2D sceneDepthTex;
// Water back-face depth texture for volume-based bump modulation (set 2)
layout(set = 2, binding = 3) uniform sampler2D waterBackDepthTex;

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
                        float foamNoiseLacunarity,
                        float bumpAmp, float waveScale) {
    vec3 p = bary.x * inPos[0] + bary.y * inPos[1] + bary.z * inPos[2];
    vec3 n = normalize(bary.x * inNormal[0] + bary.y * inNormal[1] + bary.z * inNormal[2]);
    float h = waterWaveDisplacement(p.xyz, animTime, foamNoiseScale, foamNoiseOctaves, foamNoisePersistence, foamNoiseLacunarity, bumpAmp, waveScale);
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
    
    // Select per-patch brushIndex from compressed TCS outputs (tc_fragBrushIndex / tc_fragTexWeights)
    ivec3 texIndices = max(tc_fragBrushIndex[0], ivec3(0));
    vec3 weights = tc_fragTexWeights[0] * bary.x + tc_fragTexWeights[1] * bary.y + tc_fragTexWeights[2] * bary.z;
    int chosenIdx = texIndices.x;
    if (texIndices.y >= 0 && weights.y > weights.x) chosenIdx = texIndices.y;
    if (texIndices.z >= 0 && weights.z > max(weights.x, weights.y)) chosenIdx = texIndices.z;
    if (chosenIdx < 0) chosenIdx = 0;
    // Expose the chosen brushIndex to the fragment stage
    fragBrushIndex = chosenIdx;

    // Load selected WaterParams from SSBO
    WaterParamsGPU wp = waterParams[chosenIdx];

    // Get water and noise parameters from selected params
    float time = waterRenderUBO.timeParams.x;
    // waveScale is no longer in passParams (z/w now carry nearPlane/farPlane).
    // Displacement magnitude is fully controlled by bumpAmp from the widget.
    float waveScale = 1.0;
    float noiseTimeSpeed = wp.params3.x;

    // Noise params from params2 (same as fragment shader)
    float noiseScale = wp.params2.y;
    int noiseOctaves = int(max(wp.params2.z, 1.0));
    float noisePersistence = wp.params2.w;
    float noiseLacunarity = wp.params3.y;

    float bumpAmp = wp.waveParams.z; // bump amplitude provided via Water widget

    // --- Screen-space UV of the undisplaced base vertex ---
    // Shared by the shallow-wave attenuation and the volume-based bump
    // modulation, and forwarded to the fragment stage (fragBasePos) so the
    // per-fragment analytic normal samples the SAME height field the displaced
    // geometry was built from.
    vec2 screenUV = vec2(0.0);
    float baseClipDepth = 0.0;
    bool haveScreen = false;
    {
        vec4 preClip = ubo.viewProjection * vec4(pos, 1.0);
        if (preClip.w > 0.001) {
            screenUV = clamp(preClip.xy / preClip.w * 0.5 + 0.5, 0.001, 0.999);
            baseClipDepth = preClip.z / preClip.w;
            haveScreen = true;
        }
    }

    // --- Depth-based wave attenuation (shallow: suppress waves) ---
    // Where the water surface is close to solid geometry (shallow), waves are
    // suppressed.  waveDepthTransition controls the ramp distance.
    float waveDepthTransition = wp.shallowColor.w;
    if (waveDepthTransition > 0.0 && haveScreen) {
        float solidDepthRaw = texture(sceneDepthTex, screenUV).r;
        float solidDepthLin = linearizeDepth(solidDepthRaw);
        float waterDepthLin = linearizeDepth(baseClipDepth);
        float depthDiff = max(solidDepthLin - waterDepthLin, 0.0);
        bumpAmp *= smoothstep(0.0, waveDepthTransition, depthDiff);
    }

    // --- Volume-based bump amplitude (deep water: amplify waves) ---
    // Reconstruct water thickness the same way the fragment stage does, so the
    // amplitude that drives the displaced geometry matches the one used for the
    // per-fragment shading normal.
    float volumeBumpRate = wp.reserved2.z;
    if (volumeBumpRate > 0.0 && haveScreen) {
        float backFaceDepthRaw = texture(waterBackDepthTex, screenUV).r;
        float sceneDepthRaw = texture(sceneDepthTex, screenUV).r;

        mat4 invVP = ubo.invViewProjection;
        vec4 backFaceWorldH = invVP * vec4(screenUV * 2.0 - 1.0, backFaceDepthRaw, 1.0);
        vec3 backFaceWorld = backFaceWorldH.xyz / backFaceWorldH.w;
        vec4 sceneWorldH = invVP * vec4(screenUV * 2.0 - 1.0, sceneDepthRaw, 1.0);
        vec3 sceneWorldPos = sceneWorldH.xyz / sceneWorldH.w;

        vec3 worldFrontPos = pos;
        vec3 worldRayDir = normalize(worldFrontPos - ubo.viewPos.xyz);
        float backFaceThickness = max(dot(backFaceWorld - worldFrontPos, worldRayDir), 0.0);
        float sceneThickness    = max(dot(sceneWorldPos - worldFrontPos, worldRayDir), 0.0);
        const float kMinVolumeThickness = 0.05;
        bool hasValidBackFace = (backFaceDepthRaw < 0.9999) && (backFaceThickness > kMinVolumeThickness);
        float waterThickness = hasValidBackFace ? min(backFaceThickness, sceneThickness) : sceneThickness;

        bumpAmp *= (1.0 - exp(-waterThickness * volumeBumpRate));
    }

    // Calculate wave displacement and its analytic spatial gradient using 4D
    // Perlin FBM.  A single noise evaluation yields both the height and the
    // gradient, replacing the previous 5-evaluation central-difference scheme.
    float animTime = time * noiseTimeSpeed;
    vec3 xyz = pos.xyz;

    // Surface basis (tangent plane) for projecting the analytic gradient.
    vec3 upVec = abs(normal.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(upVec, normal));
    vec3 B = cross(normal, T);

    vec4 wave = waterWaveSample(
        xyz,
        animTime,
        noiseScale,
        noiseOctaves,
        noisePersistence,
        noiseLacunarity,
        bumpAmp,
        waveScale
    );

    float waveDisplacement = wave.x;

    // Project the analytic gradient onto the tangent basis to get the
    // height-field slopes along T and B.
    float dhdT = dot(wave.yzw, T);
    float dhdB = dot(wave.yzw, B);

    vec3 bumpedN = normalize(normal - dhdT * T - dhdB * B);
    if (dot(bumpedN, normal) < 0.0) bumpedN = -bumpedN;

    // Displace along the FLAT base normal, not the perturbed one. The analytic
    // normal `bumpedN = N - dHdT*T - dHdB*B` is the exact surface normal only for
    // a height field defined as `base + N * h`. Displacing along the tilted
    // `bumpedN` instead would build a different surface whose true normal no
    // longer matches `bumpedN`, so lighting would disagree with the geometry
    // (visible especially on steep/large waves). Keeping the displacement axis
    // fixed at `normal` makes the rasterized surface and the shading normal
    // consistent.
    pos += waveDisplacement * normal;
    fragNormal = bumpedN;
    fragBaseNormal = normal;   // undisplaced (flat) interpolated base normal
    fragBasePos = vec4(pos - waveDisplacement * normal, bumpAmp);  // base pos + final amplitude

    // Debug: encode displacement as color (normalized)
    float maxExpected = bumpAmp * waveScale * 1.5; // heuristic normalization factor
    float normDisp = clamp((waveDisplacement / maxExpected) * 0.5 + 0.5, 0.0, 1.0);
    fragDebug = vec3(normDisp);
    
    fragPos = pos;
    fragPosWorld = pos;
    fragPosLightSpace = ubo.lightSpaceMatrix * vec4(pos, 1.0);
    vec4 clipPos = ubo.viewProjection * vec4(pos, 1.0);
    fragPosClip = clipPos;
    gl_Position = clipPos;
}

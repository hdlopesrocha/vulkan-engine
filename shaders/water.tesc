#version 450

#include "includes/locations.glsl"

// Water tessellation control shader — noise-adaptive tessellation
// Uses the same Perlin FBM noise function as the displacement/bump to
// allocate more triangles in areas with higher wave activity.

layout(vertices = 3) out;

layout(location = VARY_LOCALPOS) in vec3 inPos[];
layout(location = VARY_NORMAL) in vec3 inNormal[];
layout(location = VARY_UV) in vec2 inTexCoord[];
layout(location = VARY_BRUSHPATCH) flat in int pc_inBrushIndex[];

layout(location = VARY_LOCALPOS) out vec3 outPos[];
layout(location = VARY_NORMAL) out vec3 outNormal[];
layout(location = VARY_UV) out vec2 outTexCoord[];
layout(location = VARY_BRUSHPATCH) flat out ivec3 tc_fragBrushIndex[];
layout(location = VARY_TEXWEIGHTS) out vec3 tc_fragTexWeights[];

#include "includes/ubo.glsl"
#include "includes/perlin.glsl"
#include "includes/water_noise.glsl"

void main() {
    outPos[gl_InvocationID] = inPos[gl_InvocationID];
    outNormal[gl_InvocationID] = inNormal[gl_InvocationID];
    outTexCoord[gl_InvocationID] = inTexCoord[gl_InvocationID];

    int i0 = pc_inBrushIndex[0];
    int i1 = pc_inBrushIndex[1];
    int i2 = pc_inBrushIndex[2];

    int u0 = i0;
    int u1 = (i1 == u0) ? -1 : i1;
    int u2;
    if (i2 == u0 || (u1 != -1 && i2 == u1)) u2 = -1;
    else u2 = i2;

    tc_fragBrushIndex[gl_InvocationID] = ivec3(u0, u1, u2);

    int myIdx = pc_inBrushIndex[gl_InvocationID];
    vec3 texWeights = vec3(0.0);
    if (myIdx == u0) texWeights.x = 1.0;
    else if (myIdx == u1) texWeights.y = 1.0;
    else if (myIdx == u2) texWeights.z = 1.0;

    tc_fragTexWeights[gl_InvocationID] = texWeights;

    if (gl_InvocationID == 0) {
        vec3 center = (inPos[0] + inPos[1] + inPos[2]) / 3.0;
        float dist = length(ubo.viewPos.xyz - center);

        // Select the water params from the first brush index on the patch
        int idx = max(pc_inBrushIndex[0], 0);
        WaterParamsGPU wp = waterParams[idx];

        // Tessellation range from per-water-layer params
        float nearDist  = wp.tessParams.x;
        float farDist   = wp.tessParams.y;
        float minLevel  = wp.tessParams.z;
        float maxLevel  = wp.tessParams.w;
        float noiseInf  = wp.waveParams.x;

        // Base distance-based tessellation
        float distTess = clamp(farDist / max(dist, 1.0), minLevel, maxLevel);

        // Noise-adaptive modulation: evaluate the same wave displacement
        // function used by the TES and fragment shader at the patch center.
        // Higher wave activity → more tessellation.
        float timeVal = waterRenderUBO.timeParams.x * wp.params3.x;
        float lacunarity = wp.params3.y;
        float noiseVal = waterWaveDisplacement(
            center, timeVal,
            wp.params2.y, int(max(wp.params2.z, 1.0)), wp.params2.w, lacunarity,
            1.0, 1.0
        );

        // Map noise from [0,1] to a modulation factor around 1.0
        // At full noiseInfluence (1.0): range = [0.75, 1.25] → ±25%
        float noiseMod = 1.0 + noiseInf * (noiseVal - 0.5);
        float tessLevel = clamp(distTess * noiseMod, minLevel, maxLevel);

        gl_TessLevelOuter[0] = tessLevel;
        gl_TessLevelOuter[1] = tessLevel;
        gl_TessLevelOuter[2] = tessLevel;
        gl_TessLevelInner[0] = tessLevel;
    }
}

#version 450

layout(vertices = 3) out;

#include "includes/ubo.glsl"

// Pass through per-vertex varyings from vertex shader
layout(location = 0) in vec3 pc_inFragColor[];
layout(location = 1) in vec2 pc_inUV[];
layout(location = 2) in vec3 pc_inNormal[];
layout(location = 4) in vec3 pc_inPosWorld[];
layout(location = 5) in int pc_inTexIndex[];
layout(location = 7) in vec3 pc_inLocalPos[];
layout(location = 8) in vec3 pc_inLocalNormal[];


layout(location = 0) out vec3 tc_fragColor[];
layout(location = 1) out vec2 tc_fragUV[];
layout(location = 2) out vec3 tc_fragNormal[];
layout(location = 4) out vec3 tc_fragPosWorld[];
layout(location = 5) flat out ivec3 tc_fragTexIndex[];
layout(location = 11) out vec3 tc_fragTexWeights[];
layout(location = 7) out vec3 tc_fragLocalPos[];
layout(location = 8) out vec3 tc_fragLocalNormal[];


void main() {
    // Pass through per-vertex data to evaluation stage
    tc_fragColor[gl_InvocationID] = pc_inFragColor[gl_InvocationID];
    tc_fragUV[gl_InvocationID] = pc_inUV[gl_InvocationID];
    tc_fragNormal[gl_InvocationID] = pc_inNormal[gl_InvocationID];
    tc_fragPosWorld[gl_InvocationID] = pc_inPosWorld[gl_InvocationID];


    // Compress the patch's texture indices into up to three unique slots
    int i0 = pc_inTexIndex[0];
    int i1 = pc_inTexIndex[1];
    int i2 = pc_inTexIndex[2];

    int u0 = i0;
    int u1 = (i1 == u0) ? -1 : i1;
    int u2;
    if (i2 == u0 || (u1 != -1 && i2 == u1)) u2 = -1;
    else u2 = i2;

    // Store the unique indices (use -1 for empty slots)
    tc_fragTexIndex[gl_InvocationID] = ivec3(u0, u1, u2);

    // Map this corner's barycentric basis into the matching unique-slot
    int myIdx = pc_inTexIndex[gl_InvocationID];
    vec3 texWeights = vec3(0.0);
    if (myIdx == u0) texWeights.x = 1.0;
    else if (myIdx == u1) texWeights.y = 1.0;
    else if (myIdx == u2) texWeights.z = 1.0;

    tc_fragTexWeights[gl_InvocationID] = texWeights;
    tc_fragLocalPos[gl_InvocationID] = pc_inLocalPos[gl_InvocationID];
    tc_fragLocalNormal[gl_InvocationID] = pc_inLocalNormal[gl_InvocationID];
    // tangents are computed in the fragment shader for triplanar mapping

    // Compute tessellation level on GPU based on camera distance (adaptive)
    int patchTexIndex = pc_inTexIndex[0];
    float materialLevel = materials[patchTexIndex].mappingParams.y;
    bool tessEnabled = ubo.passParams.y > 0.5 && (materials[patchTexIndex].mappingParams.x > 0.5);


    // Compute patch center in world space and distance to camera
    vec3 p0 = pc_inPosWorld[0];
    vec3 p1 = pc_inPosWorld[1];
    vec3 p2 = pc_inPosWorld[2];
    vec3 center = (p0 + p1 + p2) / 3.0;
    float dist = length(center - ubo.viewPos.xyz);

    // Read tuning parameters from UBO: x=nearDist, y=farDist, z=minLevel, w=maxLevel
    float nearDist = ubo.tessParams.x;
    float farDist  = ubo.tessParams.y;
    float minLevel = ubo.tessParams.z;
    float maxLevel = ubo.tessParams.w;

    float outer;
    float inner;
    // If global tessellation is disabled, or mapping is not enabled for this patch, force tessellation to 1 (no subdivision)
    if (!tessEnabled) {
        outer = 1.0;
        inner = 1.0;
    } else {
        // Compute an adaptive tess value in [minLevel, maxLevel] based on camera distance
        float factor = clamp(1.0 - smoothstep(nearDist, farDist, dist), 0.0, 1.0); // 1.0 at near, 0.0 at far
        float tess = mix(minLevel, maxLevel, factor) + materialLevel;
        outer = tess;
        inner = tess;
    }

    gl_TessLevelOuter[0] = outer;
    gl_TessLevelOuter[1] = outer;
    gl_TessLevelOuter[2] = outer;
    gl_TessLevelInner[0] = inner;
}
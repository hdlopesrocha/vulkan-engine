#version 450

layout(vertices = 3) out;

#include "includes/ubo.glsl"

// Pass through per-vertex varyings from vertex shader
layout(location = 0) in vec3 pc_inFragColor[];
layout(location = 1) in vec2 pc_inUV[];
layout(location = 2) in vec3 pc_inNormal[];
layout(location = 4) in vec3 pc_inPosWorld[];
layout(location = 5) in float pc_inTexIndex[];
layout(location = 9) in vec4 pc_inTangent[];
layout(location = 7) in vec3 pc_inLocalPos[];
layout(location = 8) in vec3 pc_inLocalNormal[];

layout(location = 0) out vec3 tc_fragColor[];
layout(location = 1) out vec2 tc_fragUV[];
layout(location = 2) out vec3 tc_fragNormal[];
layout(location = 4) out vec3 tc_fragPosWorld[];
layout(location = 5) flat out int tc_fragTexIndex[];
layout(location = 7) out vec3 tc_fragLocalPos[];
layout(location = 8) out vec3 tc_fragLocalNormal[];
layout(location = 9) out vec4 tc_fragTangent[];

void main() {
    // Pass through per-vertex data to evaluation stage
    tc_fragColor[gl_InvocationID] = pc_inFragColor[gl_InvocationID];
    tc_fragUV[gl_InvocationID] = pc_inUV[gl_InvocationID];
    tc_fragNormal[gl_InvocationID] = pc_inNormal[gl_InvocationID];
    tc_fragNormal[gl_InvocationID] = pc_inNormal[gl_InvocationID];
    tc_fragPosWorld[gl_InvocationID] = pc_inPosWorld[gl_InvocationID];
    tc_fragTexIndex[gl_InvocationID] = int(pc_inTexIndex[gl_InvocationID] + 0.5);
    tc_fragLocalPos[gl_InvocationID] = pc_inLocalPos[gl_InvocationID];
    tc_fragLocalNormal[gl_InvocationID] = pc_inLocalNormal[gl_InvocationID];
    tc_fragTangent[gl_InvocationID] = pc_inTangent[gl_InvocationID];
    // local tangent removed

    // Compute tessellation level on GPU based on camera distance (adaptive)
    int patchTexIndex = int(pc_inTexIndex[0] + 0.5);
    // mappingParams.x == mappingEnabled (0/1). mappingParams.y stores a per-material base/max tess level.
    bool mappingEnabled = (materials[patchTexIndex].mappingParams.x > 0.5);
    float materialLevel = materials[patchTexIndex].mappingParams.y;


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

    // Compute an adaptive tess value in [minLevel, maxLevel] based on camera distance
    float factor = clamp(1.0 - smoothstep(nearDist, farDist, dist), 0.0, 1.0); // 1.0 at near, 0.0 at far
    float tess = mix(minLevel, maxLevel, factor)+materialLevel;

    float outer = tess;
    float inner = tess;
    gl_TessLevelOuter[0] = outer;
    gl_TessLevelOuter[1] = outer;
    gl_TessLevelOuter[2] = outer;
    gl_TessLevelInner[0] = inner;
}

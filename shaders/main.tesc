#version 450

layout(vertices = 3) out;

#include "includes/ubo.glsl"
#include "includes/locations.glsl"

// Pass through per-vertex varyings from vertex shader
layout(location = VARY_COLOR) in vec3 pc_inFragColor[];
layout(location = VARY_UV) in vec2 pc_inUV[];
layout(location = VARY_NORMAL) in vec3 pc_inNormal[];
layout(location = VARY_POSWORLD) in vec3 pc_inPosWorld[];
layout(location = VARY_BRUSHPATCH) flat in int pc_inBrushIndex[];
layout(location = VARY_LOCALPOS) in vec3 pc_inLocalPos[];
layout(location = VARY_LOCALNORMAL) in vec3 pc_inLocalNormal[];


layout(location = VARY_COLOR) out vec3 tc_fragColor[];
layout(location = VARY_UV) out vec2 tc_fragUV[];
layout(location = VARY_NORMAL) out vec3 tc_fragNormal[];
layout(location = VARY_POSWORLD) out vec3 tc_fragPosWorld[];
layout(location = VARY_BRUSHPATCH) flat out ivec3 tc_fragBrushIndex[];
layout(location = VARY_LOCALPOS) out vec3 tc_fragLocalPos[];
layout(location = VARY_LOCALNORMAL) out vec3 tc_fragLocalNormal[];
layout(location = VARY_TEXWEIGHTS) out vec3 tc_fragTexWeights[];


// Compute per-edge tess factor using nearest endpoint distance to camera
float computeEdgeTess(vec3 a, vec3 b, float nearDist, float farDist, float minLevel, float maxLevel, float materialLevel) {
    float da = length(a - ubo.viewPos.xyz);
    float db = length(b - ubo.viewPos.xyz);
    float d = min(da, db);
    float factor = clamp(1.0 - smoothstep(nearDist, farDist, d), 0.0, 1.0); // 1.0 at near, 0.0 at far
    return mix(minLevel, maxLevel, factor) + materialLevel * factor;
}

void main() {
    // Pass through per-vertex data to evaluation stage
    tc_fragColor[gl_InvocationID] = pc_inFragColor[gl_InvocationID];
    tc_fragUV[gl_InvocationID] = pc_inUV[gl_InvocationID];
    tc_fragNormal[gl_InvocationID] = pc_inNormal[gl_InvocationID];
    tc_fragPosWorld[gl_InvocationID] = pc_inPosWorld[gl_InvocationID];


    // Compress the patch's texture indices into up to three unique slots
    int i0 = pc_inBrushIndex[0];
    int i1 = pc_inBrushIndex[1];
    int i2 = pc_inBrushIndex[2];

    int u0 = i0;
    int u1 = (i1 == u0) ? -1 : i1;
    int u2;
    if (i2 == u0 || (u1 != -1 && i2 == u1)) u2 = -1;
    else u2 = i2;

    // Store the unique indices (use -1 for empty slots)
    tc_fragBrushIndex[gl_InvocationID] = ivec3(u0, u1, u2);

    // Map this corner's barycentric basis into the matching unique-slot
    int myIdx = pc_inBrushIndex[gl_InvocationID];
    vec3 texWeights = vec3(0.0);
    if (myIdx == u0) texWeights.x = 1.0;
    else if (myIdx == u1) texWeights.y = 1.0;
    else if (myIdx == u2) texWeights.z = 1.0;

    tc_fragTexWeights[gl_InvocationID] = texWeights;
    tc_fragLocalPos[gl_InvocationID] = pc_inLocalPos[gl_InvocationID];
    tc_fragLocalNormal[gl_InvocationID] = pc_inLocalNormal[gl_InvocationID];
    // tangents are computed in the fragment shader for triplanar mapping

    // Compute tessellation level on GPU based on camera distance (adaptive)
    int patchBrushIndex = pc_inBrushIndex[0];
    float materialLevel = materials[patchBrushIndex].mappingParams.y;
    bool tessEnabled = ubo.passParams.y > 0.5 && (materials[patchBrushIndex].mappingParams.x > 0.5);


    // Compute patch center in world space and distance to camera
    vec3 p0 = pc_inPosWorld[0];
    vec3 p1 = pc_inPosWorld[1];
    vec3 p2 = pc_inPosWorld[2];
    vec3 center = (p0 + p1 + p2) / 3.0;
    float dist = length(center - ubo.viewPos.xyz);

    // Read per-material tessellation range; near/far distances still come from the global UBO
    float nearDist = ubo.tessParams.x;
    float farDist  = ubo.tessParams.y;
    float minLevel = materials[patchBrushIndex].tessLevelParams.x;
    float maxLevel = materials[patchBrushIndex].tessLevelParams.y;

    float outer0, outer1, outer2, inner;
    if (!tessEnabled) {
        outer0 = outer1 = outer2 = 1.0;
        inner = 1.0;
    } else {
        // Map tess levels to edges: Outer0 = edge (v1,v2); Outer1 = edge (v2,v0); Outer2 = edge (v0,v1)
        outer0 = computeEdgeTess(p1, p2, nearDist, farDist, minLevel, maxLevel, materialLevel);
        outer1 = computeEdgeTess(p2, p0, nearDist, farDist, minLevel, maxLevel, materialLevel);
        outer2 = computeEdgeTess(p0, p1, nearDist, farDist, minLevel, maxLevel, materialLevel);
        // Inner level uses the max to avoid cracks across patches
        inner = max(max(outer0, outer1), outer2);
    }

    gl_TessLevelOuter[0] = outer0;
    gl_TessLevelOuter[1] = outer1;
    gl_TessLevelOuter[2] = outer2;
    gl_TessLevelInner[0] = inner;
}
#version 450

layout(vertices = 3) out;

#include "includes/ubo.glsl"

// Pass through per-vertex varyings from vertex shader
layout(location = 0) in vec3 pc_inFragColor[];
layout(location = 1) in vec2 pc_inUV[];
layout(location = 2) in vec3 pc_inNormal[];
layout(location = 3) in vec3 pc_inTangent[];
layout(location = 4) in vec3 pc_inPosWorld[];
layout(location = 5) in float pc_inTexIndex[];
layout(location = 7) in vec3 pc_inLocalPos[];
layout(location = 8) in vec3 pc_inLocalNormal[];
layout(location = 9) in vec4 pc_inLocalTangent[];

layout(location = 0) out vec3 tc_fragColor[];
layout(location = 1) out vec2 tc_fragUV[];
layout(location = 2) out vec3 tc_fragNormal[];
layout(location = 3) out vec3 tc_fragTangent[];
layout(location = 4) out vec3 tc_fragPosWorld[];
layout(location = 5) flat out int tc_fragTexIndex[];
layout(location = 7) out vec3 tc_fragLocalPos[];
layout(location = 8) out vec3 tc_fragLocalNormal[];
layout(location = 9) out vec4 tc_fragLocalTangent[];

void main() {
    // Pass through per-vertex data to evaluation stage
    tc_fragColor[gl_InvocationID] = pc_inFragColor[gl_InvocationID];
    tc_fragUV[gl_InvocationID] = pc_inUV[gl_InvocationID];
    tc_fragNormal[gl_InvocationID] = pc_inNormal[gl_InvocationID];
    tc_fragTangent[gl_InvocationID] = pc_inTangent[gl_InvocationID];
        tc_fragNormal[gl_InvocationID] = pc_inNormal[gl_InvocationID];
        tc_fragTangent[gl_InvocationID] = pc_inTangent[gl_InvocationID];
    tc_fragPosWorld[gl_InvocationID] = pc_inPosWorld[gl_InvocationID];
    tc_fragTexIndex[gl_InvocationID] = int(pc_inTexIndex[gl_InvocationID] + 0.5);
    tc_fragLocalPos[gl_InvocationID] = pc_inLocalPos[gl_InvocationID];
    tc_fragLocalNormal[gl_InvocationID] = pc_inLocalNormal[gl_InvocationID];
    tc_fragLocalTangent[gl_InvocationID] = pc_inLocalTangent[gl_InvocationID];

    // Read tessellation level from per-material mappingParams.y (set by host)
    float tessLevel = clamp(ubo.mappingParams.y, 1.0, 64.0);
    float outer = tessLevel;
    float inner = tessLevel;
    gl_TessLevelOuter[0] = outer;
    gl_TessLevelOuter[1] = outer;
    gl_TessLevelOuter[2] = outer;
    gl_TessLevelInner[0] = inner;
}

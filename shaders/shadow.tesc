#version 450

layout(vertices = 3) out;

#include "includes/ubo.glsl"

layout(location = 7) in vec3 pc_inLocalPos[];
layout(location = 8) in vec3 pc_inLocalNormal[];
layout(location = 1) in vec2 pc_inUV[];
layout(location = 5) in float pc_inTexIndex[];
layout(location = 9) in vec4 pc_inTangent[];

// Outputs to evaluation stage
layout(location = 7) out vec3 tc_fragLocalPos[];
layout(location = 8) out vec3 tc_fragLocalNormal[];
layout(location = 1) out vec2 tc_fragUV[];
layout(location = 5) flat out int tc_fragTexIndex[];
layout(location = 9) out vec4 tc_fragTangent[];

void main() {
    // Pass through necessary per-vertex local-space attributes
    tc_fragLocalPos[gl_InvocationID] = pc_inLocalPos[gl_InvocationID];
    tc_fragLocalNormal[gl_InvocationID] = pc_inLocalNormal[gl_InvocationID];
    tc_fragUV[gl_InvocationID] = pc_inUV[gl_InvocationID];
    tc_fragTexIndex[gl_InvocationID] = int(pc_inTexIndex[gl_InvocationID] + 0.5);
    tc_fragTangent[gl_InvocationID] = pc_inTangent[gl_InvocationID];

    // Set tessellation levels using mappingParams.y
    float tessLevel = clamp(ubo.mappingParams.y, 1.0, 64.0);
    gl_TessLevelOuter[0] = tessLevel;
    gl_TessLevelOuter[1] = tessLevel;
    gl_TessLevelOuter[2] = tessLevel;
    gl_TessLevelInner[0] = tessLevel;

    // Emit control point positions unchanged (positions come from vertex shader pipeline)
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
}

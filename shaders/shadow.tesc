#version 450

layout(vertices = 3) out;

// Minimal UBO for tessellation control stage (must match host layout where used)
layout(binding = 0) uniform UBO {
    mat4 mvp;
    mat4 model;
    vec4 viewPos;
    vec4 lightDir;
    vec4 lightColor;
    vec4 pomParams;
    vec4 pomFlags;
    vec4 parallaxLOD;
    vec4 mappingParams; // x=mappingMode, y=tessLevel, z=invertHeight, w=tessHeightScale
    vec4 specularParams;
    mat4 lightSpaceMatrix;
    vec4 shadowEffects;
} ubo;

// Inputs from vertex shader (match triangle.tesc convention)
layout(location = 7) in vec3 pc_inLocalPos[];
layout(location = 8) in vec3 pc_inLocalNormal[];
layout(location = 1) in vec2 pc_inUV[];
layout(location = 5) in float pc_inTexIndex[];

// Outputs to evaluation stage
layout(location = 7) out vec3 tc_fragLocalPos[];
layout(location = 8) out vec3 tc_fragLocalNormal[];
layout(location = 1) out vec2 tc_fragUV[];
layout(location = 5) flat out int tc_fragTexIndex[];

void main() {
    // Pass through necessary per-vertex local-space attributes
    tc_fragLocalPos[gl_InvocationID] = pc_inLocalPos[gl_InvocationID];
    tc_fragLocalNormal[gl_InvocationID] = pc_inLocalNormal[gl_InvocationID];
    tc_fragUV[gl_InvocationID] = pc_inUV[gl_InvocationID];
    tc_fragTexIndex[gl_InvocationID] = int(pc_inTexIndex[gl_InvocationID] + 0.5);

    // Set tessellation levels using mappingParams.y
    float tessLevel = clamp(ubo.mappingParams.y, 1.0, 64.0);
    gl_TessLevelOuter[0] = tessLevel;
    gl_TessLevelOuter[1] = tessLevel;
    gl_TessLevelOuter[2] = tessLevel;
    gl_TessLevelInner[0] = tessLevel;

    // Emit control point positions unchanged (positions come from vertex shader pipeline)
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
}

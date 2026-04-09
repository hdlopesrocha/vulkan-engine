#version 450

// Water tessellation control shader

layout(vertices = 3) out;

layout(location = 0) in vec3 inPos[];
layout(location = 1) in vec3 inNormal[];
layout(location = 2) in vec2 inTexCoord[];
layout(location = 7) flat in int pc_inTexIndex[];

layout(location = 0) out vec3 outPos[];
layout(location = 1) out vec3 outNormal[];
layout(location = 2) out vec2 outTexCoord[];
layout(location = 5) flat out ivec3 tc_fragTexIndex[];
layout(location = 11) out vec3 tc_fragTexWeights[];

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
    mat4 lightSpaceMatrix1; // cascade 1
    mat4 lightSpaceMatrix2; // cascade 2
} ubo;

void main() {
    // Pass through vertex data
    outPos[gl_InvocationID] = inPos[gl_InvocationID];
    outNormal[gl_InvocationID] = inNormal[gl_InvocationID];
    outTexCoord[gl_InvocationID] = inTexCoord[gl_InvocationID];
    
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

    if (gl_InvocationID == 0) {
        // Calculate tessellation level based on distance to camera
        vec3 center = (inPos[0] + inPos[1] + inPos[2]) / 3.0;
        float dist = length(ubo.viewPos.xyz - center);
        
        // Higher tessellation when closer
        float tessLevel = clamp(200.0 / dist, 1.0, 16.0);
        
        gl_TessLevelOuter[0] = tessLevel;
        gl_TessLevelOuter[1] = tessLevel;
        gl_TessLevelOuter[2] = tessLevel;
        gl_TessLevelInner[0] = tessLevel;
    }
}

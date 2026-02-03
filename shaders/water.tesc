#version 450

// Water tessellation control shader

layout(vertices = 3) out;

layout(location = 0) in vec3 inPos[];
layout(location = 1) in vec3 inNormal[];
layout(location = 2) in vec2 inTexCoord[];

layout(location = 0) out vec3 outPos[];
layout(location = 1) out vec3 outNormal[];
layout(location = 2) out vec2 outTexCoord[];

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

void main() {
    // Pass through vertex data
    outPos[gl_InvocationID] = inPos[gl_InvocationID];
    outNormal[gl_InvocationID] = inNormal[gl_InvocationID];
    outTexCoord[gl_InvocationID] = inTexCoord[gl_InvocationID];
    
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

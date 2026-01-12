#version 450

// Water vertex shader

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inNormal;
layout(location = 5) in int inTexIndex;

layout(location = 0) out vec3 fragPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragTexCoord;
layout(location = 3) out vec4 fragPosClip;  // clip-space position for depth lookup

layout(set = 1, binding = 0) uniform UniformBufferObject {
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
    // Pass world position directly (no model matrix needed for terrain-style water)
    fragPos = inPosition;
    fragNormal = inNormal;
    fragTexCoord = inTexCoord;
    
    vec4 clipPos = ubo.viewProjection * vec4(inPosition, 1.0);
    fragPosClip = clipPos;
    gl_Position = clipPos;
}

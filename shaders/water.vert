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
layout(location = 5) out vec3 fragPosWorld;  // world-space position for shadow cascades
layout(location = 6) out vec4 fragPosLightSpace; // light-space pos (cascade 0)
layout(location = 7) flat out int fragTexIndex;

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
    fragPos = inPosition;
    fragPosWorld = inPosition;
    fragNormal = inNormal;
    fragTexCoord = inTexCoord;
    fragPosLightSpace = ubo.lightSpaceMatrix * vec4(inPosition, 1.0);
    
    vec4 clipPos = ubo.viewProjection * vec4(inPosition, 1.0);
    fragPosClip = clipPos;
    fragTexIndex = inTexIndex;
    gl_Position = clipPos;
}

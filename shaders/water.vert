#version 450

#include "includes/locations.glsl"

// Water vertex shader

layout(location = ATTR_POS) in vec3 inPosition;
layout(location = ATTR_COLOR) in vec3 inColor;
layout(location = ATTR_UV) in vec2 inTexCoord;
layout(location = ATTR_NORMAL) in vec3 inNormal;
layout(location = ATTR_BRUSH_INDEX) in int inBrushIndex;

layout(location = VARY_LOCALPOS) out vec3 fragPos;
layout(location = VARY_NORMAL) out vec3 fragNormal;
layout(location = VARY_SHARPNORMAL) out vec3 fragBaseNormal;  // undisplaced base normal (for per-fragment detail)
layout(location = VARY_UV) out vec2 fragTexCoord;
layout(location = VARY_POSCLIP) out vec4 fragPosClip;  // clip-space position for depth lookup
layout(location = VARY_POSWORLD) out vec3 fragPosWorld;  // world-space position for shadow cascades
layout(location = VARY_POSLIGHT) out vec4 fragPosLightSpace; // light-space pos (cascade 0)
layout(location = VARY_BRUSHPATCH) flat out int fragBrushIndex;

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
    fragBaseNormal = inNormal;
    fragTexCoord = inTexCoord;
    fragPosLightSpace = ubo.lightSpaceMatrix * vec4(inPosition, 1.0);
    
    vec4 clipPos = ubo.viewProjection * vec4(inPosition, 1.0);
    fragPosClip = clipPos;
    fragBrushIndex = inBrushIndex;
    gl_Position = clipPos;
}

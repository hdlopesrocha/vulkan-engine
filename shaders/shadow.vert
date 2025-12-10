#version 450

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    mat4 model;
    vec4 viewPos;
    vec4 pomParams; // x=heightScale, y=minLayers, z=maxLayers, w=enabled
    vec4 pomFlags;  // x=flipNormalY, y=flipTangentHandedness, z=unused, w=flipParallaxDirection
} push;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inTangent;

layout(location = 0) out vec2 fragUV;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragTangent;
layout(location = 3) out vec3 fragPosWorld;

void main() {
    vec4 worldPos = push.model * vec4(inPosition, 1.0);
    fragPosWorld = worldPos.xyz;
    fragUV = inTexCoord;
    fragNormal = mat3(push.model) * inNormal;
    fragTangent = mat3(push.model) * inTangent;
    
    // MVP already includes model transform, apply to local position
    gl_Position = push.mvp * vec4(inPosition, 1.0);
}

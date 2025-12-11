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

layout(location = 0) out vec2 pc_inUV;
layout(location = 1) out vec3 pc_inNormal;
layout(location = 3) out vec3 pc_inPosWorld;
layout(location = 4) out vec3 pc_inTangent;
layout(location = 5) out float pc_inTexIndex;
layout(location = 7) out vec3 pc_inLocalPos;
layout(location = 8) out vec3 pc_inLocalNormal;
layout(location = 9) out vec3 pc_inLocalTangent;

void main() {
    vec4 worldPos = push.model * vec4(inPosition, 1.0);
    pc_inPosWorld = worldPos.xyz;
    pc_inUV = inTexCoord;
    pc_inNormal = mat3(push.model) * inNormal;
    pc_inTangent = mat3(push.model) * inTangent;
    pc_inTexIndex = 0.0; // default single-layer
    pc_inLocalPos = inPosition;
    pc_inLocalNormal = inNormal;
    pc_inLocalTangent = inTangent;
    
    // MVP already includes model transform, apply to local position
    gl_Position = push.mvp * vec4(inPosition, 1.0);
}

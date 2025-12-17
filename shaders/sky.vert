#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 fragPosWorld;
layout(location = 1) out vec3 fragNormal;

#include "includes/ubo.glsl"

void main() {
    // Center sphere at camera position (UBO exposes viewPos)
    vec3 worldPos = inPosition * 50.0 + ubo.viewPos.xyz; // Large radius
    fragPosWorld = worldPos;
    fragNormal = inNormal;
    gl_Position = ubo.viewProjection * vec4(worldPos, 1.0);
}

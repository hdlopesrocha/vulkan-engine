#version 450

#include "includes/locations.glsl"

layout(location = ATTR_POS) in vec3 inPosition;
layout(location = ATTR_NORMAL) in vec3 inNormal;

layout(location = VARY_POSWORLD) out vec3 fragPosWorld;
layout(location = VARY_NORMAL) out vec3 fragNormal;

#include "includes/ubo.glsl"

void main() {
    // Center sphere at camera position (UBO exposes viewPos)
    vec3 worldPos = inPosition * 50.0 + ubo.viewPos.xyz; // Large radius
    fragPosWorld = worldPos;
    fragNormal = inNormal;
    gl_Position = ubo.viewProjection * vec4(worldPos, 1.0);
}

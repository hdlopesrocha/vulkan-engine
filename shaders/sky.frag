#version 450

layout(location = 0) in vec3 fragPosWorld;
layout(location = 1) in vec3 fragNormal;

#include "includes/ubo.glsl"

layout(location = 0) out vec4 outColor;

void main() {
    // Compute direction from camera to fragment (UBO exposes viewPos)
    vec3 viewDir = normalize(fragPosWorld - ubo.viewPos.xyz);
    // Use the Y component for gradient (up = 1, down = -1)
    float t = clamp(viewDir.y * 0.5 + 0.5, 0.0, 1.0);
    // Define gradient colors (zenith and horizon)
    vec3 horizonColor = vec3(0.6, 0.7, 0.9); // light blue
    vec3 zenithColor = vec3(0.05, 0.15, 0.4); // deep blue
    vec3 color = mix(horizonColor, zenithColor, t);
    outColor = vec4(color, 1.0);
}

#version 450

// Depth-only pass for water back-face rendering.
// Uses the same vertex + tessellation shaders as the main water pass.
// Only depth is written by the rasterizer; no color output needed.

// Declare inputs that match the tessellation evaluation shader outputs
// to keep the SPIR-V interface consistent and avoid validation warnings.
layout(location = 0) in vec3 fragPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in vec4 fragPosClip;
layout(location = 4) in vec3 fragDebug;
layout(location = 5) in vec3 fragPosWorld;
layout(location = 6) in vec4 fragPosLightSpace;
layout(location = 7) flat in int fragTexIndex;

void main() {
    // Intentionally empty — depth is recorded automatically.
}

#version 450

#include "includes/locations.glsl"

// Depth-only pass for water back-face rendering.
// Uses the same vertex + tessellation shaders as the main water pass.
// Only depth is written by the rasterizer; no color output needed.

// Declare inputs that match the tessellation evaluation shader outputs
// to keep the SPIR-V interface consistent and avoid validation warnings.
layout(location = VARY_LOCALPOS) in vec3 fragPos;
layout(location = VARY_NORMAL) in vec3 fragNormal;
layout(location = VARY_UV) in vec2 fragTexCoord;
layout(location = VARY_POSCLIP) in vec4 fragPosClip;
layout(location = VARY_DEBUG) in vec3 fragDebug;
layout(location = VARY_POSWORLD) in vec3 fragPosWorld;
layout(location = VARY_POSLIGHT) in vec4 fragPosLightSpace;
layout(location = VARY_BRUSHPATCH) flat in int fragBrushIndex;

void main() {
    // Intentionally empty — depth is recorded automatically.
}

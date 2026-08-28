#version 450

#include "includes/locations.glsl"

// Back-face depth pass: writes water back-face depth. This pass no longer
// samples the solid scene depth — water is decoupled from the solid pass, so
// back-faces are not clipped against solids here. Water-vs-solid occlusion is
// resolved later at the composite stage (postprocess.frag), which discards
// water fragments that are behind the scene depth.

layout(location = VARY_LOCALPOS) in vec3 fragPos;
layout(location = VARY_NORMAL) in vec3 fragNormal;
layout(location = VARY_UV) in vec2 fragTexCoord;
layout(location = VARY_POSCLIP) in vec4 fragPosClip;
layout(location = VARY_DEBUG) in vec3 fragDebug;
layout(location = VARY_POSWORLD) in vec3 fragPosWorld;
layout(location = VARY_POSLIGHT) in vec4 fragPosLightSpace;
layout(location = VARY_BRUSHPATCH) flat in int fragBrushIndex;

void main() {
    // No scene-depth discard: water is independent of the solid pass. The
    // depth attachment still receives the back-face depth, which the water
    // geometry pass samples for refraction/subsurface shading.
}

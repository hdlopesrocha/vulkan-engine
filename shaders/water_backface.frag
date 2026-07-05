#version 450

#include "includes/locations.glsl"

// Back-face depth pass: writes water back-face depth, clipping fragments
// that are behind the scene by sampling the scene depth texture.

layout(set = 2, binding = 1) uniform sampler2D sceneDepthTex;

layout(location = VARY_LOCALPOS) in vec3 fragPos;
layout(location = VARY_NORMAL) in vec3 fragNormal;
layout(location = VARY_UV) in vec2 fragTexCoord;
layout(location = VARY_POSCLIP) in vec4 fragPosClip;
layout(location = VARY_DEBUG) in vec3 fragDebug;
layout(location = VARY_POSWORLD) in vec3 fragPosWorld;
layout(location = VARY_POSLIGHT) in vec4 fragPosLightSpace;
layout(location = VARY_BRUSHPATCH) flat in int fragBrushIndex;

void main() {
    // Sample the scene depth at this fragment's screen position.
    // The scene depth is stored in SHADER_READ_ONLY layout — no copy needed.
    vec2 uv = gl_FragCoord.xy / vec2(textureSize(sceneDepthTex, 0));
    float sceneDepth = texture(sceneDepthTex, uv).r;

    // If this fragment is behind the scene surface, discard it.
    // Small epsilon avoids z-fighting at water-scene boundaries.
    if (gl_FragCoord.z > sceneDepth + 0.00001) discard;
}

#version 450

// Impostor EVSM2 shadow pass: uses vertex position from vertex shader
// to write EVSM moments.

#include "includes/locations.glsl"
#include "includes/evsm_write.glsl"

layout(location = VARY_POSWORLD) in vec3 inWorldPos;
layout(location = VARY_TANGENTWS) flat in vec3 inInstanceOffset;

layout(set = 0, binding = 0) uniform SolidParamsUBO {
    mat4 viewProjection;
    vec4 viewPos;
} ubo;

layout(location = FRAG_OUT_COLOR) out vec2 outEVSM;

layout(set = 2, binding = 0) uniform WindParamsUBO {
    vec4 windDirAndStrength;
    vec4 windNoise;
    vec4 windShape;
    vec4 windTurbulence;
    vec4 densityParams;
    vec4 cameraPosAndFalloff;
} windParams;

layout(push_constant) uniform PushConstants {
    float billboardScale;
    float windEnabled;
    float windTime;
    float impostorDistance;
};

void main() {
    // Dithered cross-fade (same as impostors_depth.frag).
    if (impostorDistance > 0.0) {
        float dist       = distance(ubo.viewPos.xyz, inInstanceOffset);
        float fadeAlpha  = 1.0 - smoothstep(impostorDistance * 0.50, impostorDistance * 1.15, dist);
        const int M[16]  = int[16](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);
        float threshold  = float(M[(int(gl_FragCoord.y) & 3) * 4 + (int(gl_FragCoord.x) & 3)]) / 16.0;
        if (threshold < fadeAlpha) discard;
    }

    vec4 lsPos = ubo.viewProjection * vec4(inWorldPos, 1.0);
    float depth = clamp(lsPos.z / lsPos.w, 0.0, 1.0);

    gl_FragDepth = depth;

    outEVSM = evsmMoments(depth);
}

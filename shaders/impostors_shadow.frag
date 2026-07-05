#version 450

// Impostor EVSM shadow pass: uses vertex position from vertex shader
// to write EVSM moments. Matches shadow_evsm.frag logic.

#include "includes/locations.glsl"

layout(location = VARY_POSWORLD) in vec3 inWorldPos;
layout(location = VARY_TANGENTWS) flat in vec3 inInstanceOffset;

layout(set = 0, binding = 0) uniform SolidParamsUBO {
    mat4 viewProjection;
    vec4 viewPos;
} ubo;

layout(location = FRAG_OUT_COLOR) out vec4 outEVSM;

layout(push_constant) uniform PushConstants {
    float billboardScale;
    float windEnabled;
    float windTime;
    float impostorDistance;
    vec4 windDirAndStrength;
    vec4 windNoise;
    vec4 windShape;
    vec4 windTurbulence;
    vec4 densityParams;
    vec4 cameraPosAndFalloff;
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

    // Project vertex position to light space (identical to shadow_evsm.frag).
    vec4 lsPos = ubo.viewProjection * vec4(inWorldPos, 1.0);
    float depth = clamp(lsPos.z / lsPos.w, 0.0, 1.0);

    gl_FragDepth = depth;

    // EVSM moments: exp(c*d), exp(2c*d), exp(-c*d), exp(-2c*d)
    float c = 2.0;
    float posM1 = exp( c * depth);
    float posM2 = exp( 2.0 * c * depth);
    float negM1 = exp(-c * depth);
    float negM2 = exp(-2.0 * c * depth);
    outEVSM = vec4(posM1, posM2, negM1, negM2);
}

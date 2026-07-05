#version 450

// Impostor EVSM shadow pass: reprojects captured depth into light space
// and writes EVSM moments so impostors cast shadows.

#include "includes/locations.glsl"

layout(location = VARY_UV) in vec3 inTexCoord;  // xy=UV, z=float(layerIdx)
layout(location = VARY_TANGENTWS) flat in vec3 inInstanceOffset;

layout(set = 0, binding = 0) uniform SolidParamsUBO {
    mat4 viewProjection; // light VP
    vec4 viewPos;
} ubo;

layout(set = 1, binding = 0) uniform sampler2DArray depthArray;

layout(set = 1, binding = 1) readonly buffer CaptureInvVP {
    mat4 invVP[];
};

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

layout(location = FRAG_OUT_COLOR) out vec4 outEVSM;

void main() {
    int layer = int(inTexCoord.z);

    float texDepth = texture(depthArray, inTexCoord).r;
    if (texDepth >= 1.0 || texDepth <= 0.0) discard;

    // Reconstruct world position from captured depth.
    vec2 ndc_xy = inTexCoord.xy * 2.0 - 1.0;
    vec4 clipPos = vec4(ndc_xy, texDepth, 1.0);
    vec4 worldPos = invVP[layer] * clipPos;
    worldPos /= worldPos.w;

    // Translate from capture origin to instance position.
    worldPos.xyz += inInstanceOffset;

    // Dithered cross-fade with vegetation (same as depth pass).
    if (impostorDistance > 0.0) {
        float dist       = distance(ubo.viewPos.xyz, inInstanceOffset);
        float fadeAlpha  = 1.0 - smoothstep(impostorDistance * 0.50, impostorDistance * 1.15, dist);
        const int M[16]  = int[16](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);
        float threshold  = float(M[(int(gl_FragCoord.y) & 3) * 4 + (int(gl_FragCoord.x) & 3)]) / 16.0;
        if (threshold < fadeAlpha) discard;
    }

    // Reproject to light space and compute EVSM moments.
    vec4 lsPos = ubo.viewProjection * worldPos;
    float depth = clamp(lsPos.z / lsPos.w, 0.0, 1.0);

    // Write depth for the shadow mapper's depth test.
    gl_FragDepth = depth;

    // EVSM moments: exp(c*d), exp(2c*d), exp(-c*d), exp(-2c*d)
    float c = 2.0;
    float posM1 = exp( c * depth);
    float posM2 = exp( 2.0 * c * depth);
    float negM1 = exp(-c * depth);
    float negM2 = exp(-2.0 * c * depth);
    outEVSM = vec4(posM1, posM2, negM1, negM2);
}

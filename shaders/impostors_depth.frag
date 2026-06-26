#version 450

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

    // Dithered cross-fade with vegetation (complementary to vegetation depth).
    // Only write depth for pixels that the impostor color pass would shade.
    if (impostorDistance > 0.0) {
        float dist       = distance(ubo.viewPos.xyz, inInstanceOffset);
        float fadeAlpha  = 1.0 - smoothstep(impostorDistance * 0.50, impostorDistance * 1.15, dist);
        const int M[16]  = int[16](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);
        float threshold  = float(M[(int(gl_FragCoord.y) & 3) * 4 + (int(gl_FragCoord.x) & 3)]) / 16.0;
        if (threshold < fadeAlpha) discard;  // complementary: keep where vegetation depth discards
    }

    // Reproject to camera space.
    vec4 camClipPos = ubo.viewProjection * worldPos;
    gl_FragDepth = clamp(camClipPos.z / camClipPos.w, 0.0, 1.0);
}

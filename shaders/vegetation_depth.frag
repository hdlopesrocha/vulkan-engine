#version 450

#include "includes/locations.glsl"
#include "includes/ubo.glsl"

layout(location = VARY_UV) in vec3 inTexCoord;
layout(location = VARY_BRUSHPATCH) flat in int inBrushIndex;
layout(location = VARY_POSWORLD) in vec3 inWorldPos;
layout(location = VARY_PLANE_NORMAL) flat in vec3 inPlaneNormal;
layout(location = VARY_POSLIGHT) flat in vec3 inTangentWS;

layout(set = 1, binding = 1) uniform sampler2DArray normalArray;
layout(set = 1, binding = 2) uniform sampler2DArray opacityArray;

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
    vec3 coord = vec3(inTexCoord.xy, inTexCoord.z);

    float opacity     = texture(opacityArray, coord).r;
    vec3  leafNormEnc = texture(normalArray,  coord).rgb;
    vec3  bgNormEnc   = textureLod(normalArray, coord, 5.0).rgb;

    vec3 leafNorm = normalize(leafNormEnc * 2.0 - 1.0);
    vec3 bgNorm   = normalize(bgNormEnc   * 2.0 - 1.0);

    float leafNConf = clamp(leafNorm.z, 0.0, 1.0);
    float bgNConf   = clamp(bgNorm.z,   0.0, 1.0);

    float opacityWeight = smoothstep(0.35, 0.65, opacity);
    float normalWeight  = mix(bgNConf, leafNConf, opacityWeight);
    float weight        = opacityWeight * normalWeight;

    if (weight < 0.3) discard;

    // Cross-fade with impostors: same dithering as vegetation.frag.
    if (impostorDistance > 0.0) {
        float dist       = distance(ubo.viewPos.xyz, inWorldPos);
        float fadeAlpha  = 1.0 - smoothstep(impostorDistance * 0.50, impostorDistance * 1.15, dist);
        const int M[16]  = int[16](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);
        float threshold  = float(M[(int(gl_FragCoord.y) & 3) * 4 + (int(gl_FragCoord.x) & 3)]) / 16.0;
        if (threshold >= fadeAlpha) discard;
    }

    // Depth written (default gl_FragDepth) only for fragments that survive
    // the same discard criteria as the shading pass.
}

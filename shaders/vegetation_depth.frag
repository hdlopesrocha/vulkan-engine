#version 450

#include "includes/locations.glsl"
#include "includes/ubo.glsl"
#include "includes/perlin.glsl"

layout(location = VARY_UV) in vec3 inTexCoord;
layout(location = VARY_BRUSHPATCH) flat in int inBrushIndex;
layout(location = VARY_POSWORLD) in vec3 inWorldPos;
layout(location = VARY_PLANE_NORMAL) flat in vec3 inPlaneNormal;
layout(location = VARY_POSLIGHT) flat in vec3 inTangentWS;
layout(location = VARY_ROTFRAC) in float inFireSeed; // per-instance random for fire animation

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

#include "includes/fire_common.glsl"

void main() {
    vec3 coord = vec3(inTexCoord.xy, inTexCoord.z);

    // Fire uses the same flame coverage test as the color pass so the depth
    // prepass writes exactly the fragments the shading pass keeps.
    if (inBrushIndex == FIRE_BILLBOARD_INDEX) {
        vec4 flame = evaluateFire(inTexCoord.xy, inFireSeed, windTime);
        if (flame.a < 0.35) discard;
        return;
    }

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

    // Impostor hand-off is a direct per-instance swap at impostorDistance
    // (see vegetation.vert / vegetation.frag): no dithered fade here, so the
    // depth prepass writes exactly the fragments the shading pass keeps.

    // Depth written (default gl_FragDepth) only for fragments that survive
    // the same discard criteria as the shading pass.
}

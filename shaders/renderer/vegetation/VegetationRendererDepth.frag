#version 450

#include "../../includes/Locations.glsl"
#include "../../includes/SceneBindings.glsl"
#include "../../includes/noise/Perlin.glsl"

layout(location = VARY_UV) in vec3 inTexCoord;
// M5 (perf report 28): dead inBrushIndex / inWorldPos / inPlaneNormal /
// inTangentWS declarations removed (unread below; the shared billboard VS
// writes stay — color needs them).

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

    // Cheap-first discard (perf report 26 H6): the single opacity tap is a
    // sufficient cutout test (opacityWeight < 0.3 implies weight < 0.3, since
    // normalWeight is in [0,1]), so the exact discard set is unchanged while
    // discarded pixels skip both normal fetches. textureGrad with
    // pre-discard gradients keeps the survivors' LOD bit-identical (the LOD5
    // tap was already explicit).
    float opacity       = texture(opacityArray, coord).r;
    float opacityWeight = smoothstep(0.35, 0.65, opacity);
    if (opacityWeight < 0.3) discard;

    vec2 lodGradX = dFdx(coord.xy);
    vec2 lodGradY = dFdy(coord.xy);
    vec3  leafNormEnc = textureGrad(normalArray, coord, lodGradX, lodGradY).rgb;
    vec3  bgNormEnc   = textureLod(normalArray, coord, 5.0).rgb;

    vec3 leafNorm = normalize(leafNormEnc * 2.0 - 1.0);
    vec3 bgNorm   = normalize(bgNormEnc   * 2.0 - 1.0);

    float leafNConf = clamp(leafNorm.z, 0.0, 1.0);
    float bgNConf   = clamp(bgNorm.z,   0.0, 1.0);

    float normalWeight  = mix(bgNConf, leafNConf, opacityWeight);
    float weight        = opacityWeight * normalWeight;

    if (weight < 0.3) discard;

    // Impostor hand-off is a direct per-instance swap at impostorDistance
    // (see VegetationRenderer.vert / VegetationRenderer.frag): no dithered fade here, so the
    // depth prepass writes exactly the fragments the shading pass keeps.

    // Depth written (default gl_FragDepth) only for fragments that survive
    // the same discard criteria as the shading pass.
}

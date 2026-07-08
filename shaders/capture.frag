#version 450
// Impostor capture — stores composite albedo WITHOUT baked lighting.
// Real-time ambient + diffuse + specular is applied in impostors.frag.

#include "includes/locations.glsl"

layout(location = VARY_UV) in vec3 inTexCoord;
layout(location = VARY_BRUSHPATCH) flat in int inBrushIndex;
layout(location = VARY_POSWORLD) in      vec3 inWorldPos;
layout(location = VARY_PLANE_NORMAL) flat in vec3 inPlaneNormal;
layout(location = VARY_POSLIGHT) flat in vec3 inTangentWS;

layout(location = FRAG_OUT_COLOR) out vec4 outColor;
layout(location = FRAG_OUT_NORMAL) out vec4 outNormal; // world-space normal, encoded to [0,1]
layout(location = FRAG_OUT_DEPTH) out float outDepth;  // device Z for shadow-map reprojection

layout(set = 0, binding = 0) uniform SolidParamsUBO {
    mat4 viewProjection;
    vec4 viewPos;
    vec4 lightDir;
    vec4 lightColor;
} ubo;

layout(set = 1, binding = 0) uniform sampler2DArray albedoArray;
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

    vec4  leafAlbedo  = texture(albedoArray,  coord);
    float opacity     = texture(opacityArray, coord).r;
    vec3  leafNormEnc = texture(normalArray,  coord).rgb;

    const float kAvgMip = 5.0;
    vec3 bgAlbedo  = textureLod(albedoArray, coord, kAvgMip).rgb;
    vec3 bgNormEnc = textureLod(normalArray, coord, kAvgMip).rgb;

    vec3 leafNorm = normalize(leafNormEnc * 2.0 - 1.0);
    vec3 bgNorm   = normalize(bgNormEnc   * 2.0 - 1.0);

    float leafNConf     = clamp(leafNorm.z, 0.0, 1.0);
    float bgNConf       = clamp(bgNorm.z,   0.0, 1.0);
    float opacityWeight = smoothstep(0.35, 0.65, opacity);
    float normalWeight  = mix(bgNConf, leafNConf, opacityWeight);
    float weight        = opacityWeight * normalWeight;

    if (weight < 0.5) discard;

    // Raw composite albedo — no lighting baked in.
    outColor.rgb = mix(bgAlbedo, leafAlbedo.rgb, weight);
    outColor.a   = 1.0; // cleared areas stay 0,0,0,0 → discarded at render time

    // World-space composite normal for real-time lighting in impostors.frag.
    // Build TBN from the billboard plane geometry — identical to vegetation.frag,
    // including the gl_FrontFacing flip so back-facing plane pixels get the correct
    // inward-pointing normal instead of the outward one (which would make impostors
    // appear systematically brighter than real vegetation for an overhead sun).
    vec3 T     = normalize(inTangentWS);
    vec3 faceN = gl_FrontFacing ? normalize(inPlaneNormal) : -normalize(inPlaneNormal);
    vec3 B     = normalize(cross(T, faceN));
    mat3 TBN   = mat3(T, B, faceN);
    vec3 wsLeaf = normalize(TBN * leafNorm);
    vec3 wsNorm = wsLeaf;
    outNormal   = vec4(wsNorm * 0.5 + 0.5, 1.0);
    outDepth    = gl_FragCoord.z;
}

#version 450

#include "../../includes/Locations.glsl"

layout(location = VARY_UV) in vec3 inTexCoord;  // xy=uv, z=array layer
layout(location = VARY_BRUSHPATCH) in flat int inBrushIndex;
layout(location = VARY_POSWORLD) in      vec3 inWorldPos;    // interpolated vertex world position
layout(location = VARY_PLANE_NORMAL) in flat vec3 inPlaneNormal; // billboard face normal (world space)
layout(location = VARY_POSLIGHT) in flat vec3 inTangentWS;   // billboard tangent (world space)
// H8 (perf report 27): uniform light direction already normalized per vertex.
layout(location = VARY_TANGENTWS) in flat vec3 inLightDirN;
layout(location = FRAG_OUT_COLOR) out vec4 outColor;

#include "../../includes/SceneBindings.glsl"
#include "../../includes/DebugModes.glsl"
#include "../../includes/sky/SkyView.glsl"
#include "../../includes/noise/Perlin.glsl"
#include "../../includes/sky/Clouds.glsl"

layout(set = 0, binding = 4) uniform sampler2D shadowMap;
layout(set = 0, binding = 8) uniform sampler2D shadowMap1;
layout(set = 0, binding = 9) uniform sampler2D shadowMap2;

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

vec3 fragPosWorld; // set in main() — required by shadows.glsl cascades 1 & 2

#include "../../includes/shadow/Shadows.glsl"

void main() {
    // Debug: raw LOD source, no shading. Red = billboard (this shader),
    // blue = impostor (impostors.frag). The color edge is the hand-off.
    if (ubo.debugMode == DEBUG_MODE_VEGETATION_LOD) {
        outColor = vec4(1.0, 0.0, 0.0, 1.0);
        return;
    }

    vec3 coord = vec3(inTexCoord.xy, inTexCoord.z);
    bool shadowPass = windEnabled < 0.0;
    fragPosWorld = inWorldPos; // must be set before any ShadowCalculation call

    // Cheap-first discard (perf report 26 H6): the single opacity tap drives
    // the sufficient cutout test — opacityWeight < 0.3 implies weight < 0.3
    // (normalWeight mixes two [0,1] confidences), so discarded pixels pay one
    // fetch instead of three and survivors compute the same values as the
    // old fetch-everything-first order. The albedo/normal fetches below run
    // only for survivors via textureGrad with pre-discard gradients: the
    // gradients are captured in uniform control flow, so LOD selection is
    // exactly what the old implicit fetches computed (post-discard implicit
    // derivatives would be undefined on partially-discarded quads).
    float opacity       = texture(opacityArray, coord).r;
    float opacityWeight = smoothstep(0.35, 0.65, opacity);
    if (opacityWeight < 0.3) discard;

    vec2 lodGradX = dFdx(coord.xy);
    vec2 lodGradY = dFdy(coord.xy);

    // Per-pixel leaf samples (survivors only; bit-identical values).
    vec4  leafAlbedo  = textureGrad(albedoArray, coord, lodGradX, lodGradY);
    vec3  leafNormEnc = textureGrad(normalArray,  coord, lodGradX, lodGradY).rgb;

    // Decode tangent-space normal from [0,1] to [-1,1].
    vec3 leafNorm = normalize(leafNormEnc * 2.0 - 1.0);

    // Normal confidence: Z in tangent space — 1 = facing viewer, 0 = grazing.
    float leafNConf = clamp(leafNorm.z, 0.0, 1.0);

    // Impostor hand-off is DIRECT (no dithered fade): the per-instance swap
    // at impostorDistance is documented once in VegetationRenderer.vert
    // (canonical removed-dither note, perf report 26 L15).

    // Background: nearest-leaf-filled by CPU compositor; high-mip gives spatial blend.
    const float kAvgMip = 5.0;
    vec3 bgAlbedo  = textureLod(albedoArray, coord, kAvgMip).rgb;
    vec3 bgNormEnc = textureLod(normalArray, coord, kAvgMip).rgb;

    vec3  bgNorm  = normalize(bgNormEnc * 2.0 - 1.0);
    float bgNConf = clamp(bgNorm.z,   0.0, 1.0);

    // Compositing weight: opacity sigmoid + normal confidence.
    float normalWeight = mix(bgNConf, leafNConf, opacityWeight);
    float weight       = opacityWeight * normalWeight;

    if (weight < 0.3) discard;
    // ---------------------------------------------------------------
    // Build TBN from billboard plane geometry.
    //
    // T = tangent (along billboard width, from geometry shader).
    // N = plane face normal (flipped for back-faces via gl_FrontFacing).
    // B = cross(T, N)  =>  bitangent == dWorldPos/dV (top-to-bottom).
    //
    // TBN * tangentNormal  =>  world-space normal.
    // ---------------------------------------------------------------
    vec3 T     = normalize(inTangentWS);
    vec3 faceN = gl_FrontFacing ? normalize(inPlaneNormal) : -normalize(inPlaneNormal);
    vec3 B     = normalize(cross(T, faceN));
    mat3 TBN   = mat3(T, B, faceN);

    // World-space normal derived from the per-pixel tangent-space normal map.
    vec3 worldNormal = normalize(TBN * leafNorm);

    // ---------------------------------------------------------------
    // Lighting: ambient + Lambertian diffuse + Blinn-Phong specular.
    // ubo.lightDir points FROM the light source toward the scene.
    // ---------------------------------------------------------------
    // H8 (perf report 27): L arrives normalized from the vertex stage (flat
    // varying, uniform input); the lighting chain below is unchanged.
    vec3  L     = inLightDirN;
    float NdotL = max(dot(worldNormal, L), 0.0);

    vec3  V     = normalize(ubo.viewPosition - inWorldPos);
    vec3  H     = normalize(L + V);
    float NdotH = (NdotL > 0.0) ? max(dot(worldNormal, H), 0.0) : 0.0;

    const float kAmbient  = 0.30;
    const float kSpecular = 0.08;
    const float kShine    = 16.0;
    vec3 ambient  = kAmbient            * ubo.lightColor;
    vec3 diffuse  = NdotL               * ubo.lightColor;
    vec3 specular = pow(NdotH, kShine) * kSpecular * ubo.lightColor;

    float shadow = 0.0;
    if (!shadowPass && (ubo.shadowsEnabled != 0u)) {
        if (NdotL > 0.01) {
            float bias = max(0.0005 * (1.0 - NdotL), 0.00005);
            vec4 fragPosLightSpace = ubo.lightSpaceMatrix * vec4(inWorldPos, 1.0);
            shadow = ShadowCalculationHard(fragPosLightSpace, inWorldPos, bias);
        } else {
            shadow = 1.0;
        }
    }
    // Volumetric cloud shadows on vegetation (same projection as terrain).
    if (!shadowPass && (sky.cloudsEnabled != 0u) && NdotL > 0.01) {
        float cloudShadow = cloudShadowAt(inWorldPos);
        shadow = 1.0 - (1.0 - shadow) * (1.0 - cloudShadow);
    }

    vec3 lighting = ambient + (diffuse + specular) * (1.0 - shadow);

    // Final colour: leaf over nearest-leaf background, lit by world-space normal.
    outColor.rgb = mix(bgAlbedo, leafAlbedo.rgb, weight) * lighting;
    outColor.a   = 1.0;
}

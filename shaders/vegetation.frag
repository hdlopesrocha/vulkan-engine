#version 450
layout(location = 0) in vec3 inTexCoord;  // xy=uv, z=array layer
layout(location = 1) in flat int inTexIndex;
layout(location = 2) in      vec3 inWorldPos;    // interpolated vertex world position
layout(location = 3) in flat vec3 inPlaneNormal; // billboard face normal (world space)
layout(location = 4) in flat vec3 inTangentWS;   // billboard tangent (world space)
layout(location = 0) out vec4 outColor;

#include "includes/ubo.glsl"

layout(set = 0, binding = 4) uniform sampler2D shadowMap;
layout(set = 0, binding = 8) uniform sampler2D shadowMap1;
layout(set = 0, binding = 9) uniform sampler2D shadowMap2;

layout(set = 1, binding = 0) uniform sampler2DArray albedoArray;
layout(set = 1, binding = 1) uniform sampler2DArray normalArray;
layout(set = 1, binding = 2) uniform sampler2DArray opacityArray;

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

vec3 fragPosWorld; // set in main() — required by shadows.glsl cascades 1 & 2

#include "includes/shadows.glsl"

void main() {
    vec3 coord = vec3(inTexCoord.xy, inTexCoord.z);
    bool shadowPass = windEnabled < 0.0;
    fragPosWorld = inWorldPos; // must be set before any ShadowCalculation call

    // Per-pixel samples.
    vec4  leafAlbedo  = texture(albedoArray,  coord);
    float opacity     = texture(opacityArray, coord).r;
    vec3  leafNormEnc = texture(normalArray,  coord).rgb;

    // Background: nearest-leaf-filled by CPU compositor; high-mip gives spatial blend.
    const float kAvgMip = 5.0;
    vec3 bgAlbedo  = textureLod(albedoArray, coord, kAvgMip).rgb;
    vec3 bgNormEnc = textureLod(normalArray, coord, kAvgMip).rgb;

    // Decode tangent-space normals from [0,1] to [-1,1].
    vec3 leafNorm = normalize(leafNormEnc * 2.0 - 1.0);
    vec3 bgNorm   = normalize(bgNormEnc   * 2.0 - 1.0);

    // Normal confidence: Z in tangent space — 1 = facing viewer, 0 = grazing.
    float leafNConf = clamp(leafNorm.z, 0.0, 1.0);
    float bgNConf   = clamp(bgNorm.z,   0.0, 1.0);

    // Compositing weight: opacity sigmoid + normal confidence.
    float opacityWeight = smoothstep(0.35, 0.65, opacity);
    float normalWeight  = mix(bgNConf, leafNConf, opacityWeight);
    float weight        = opacityWeight * normalWeight;

    if (weight < 0.5) discard;
    // Cross-fade with impostors: dithered fade-out in the transition zone.
    // Vegetation fades from fully opaque (at 0.85×impostorDistance) to fully gone
    // (at 1.15×impostorDistance) using complementary Bayer 4×4 ordered dithering.
    // The impostor shader uses the inverse condition, so together they cover 100% of pixels.
    if (!shadowPass && impostorDistance > 0.0) {
        float dist       = distance(ubo.viewPos.xyz, inWorldPos);
        float fadeAlpha  = 1.0 - smoothstep(impostorDistance * 0.85, impostorDistance * 1.15, dist);
        const int M[16]  = int[16](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);
        float threshold  = float(M[(int(gl_FragCoord.y) & 3) * 4 + (int(gl_FragCoord.x) & 3)]) / 16.0;
        if (threshold >= fadeAlpha) discard;
    }
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
    vec3  L     = normalize(-ubo.lightDir.xyz);
    float NdotL = max(dot(worldNormal, L), 0.0);

    vec3  V     = normalize(ubo.viewPos.xyz - inWorldPos);
    vec3  H     = normalize(L + V);
    float NdotH = (NdotL > 0.0) ? max(dot(worldNormal, H), 0.0) : 0.0;

    const float kAmbient  = 0.30;
    const float kSpecular = 0.08;
    const float kShine    = 16.0;
    vec3 ambient  = kAmbient            * ubo.lightColor.rgb;
    vec3 diffuse  = NdotL               * ubo.lightColor.rgb;
    vec3 specular = pow(NdotH, kShine) * kSpecular * ubo.lightColor.rgb;

    float shadow = 0.0;
    if (!shadowPass && ubo.shadowEffects.w > 0.5) {
        if (NdotL > 0.01) {
            float bias = max(0.003 * (1.0 - NdotL), 0.001);
            vec4 fragPosLightSpace = ubo.lightSpaceMatrix * vec4(inWorldPos, 1.0);
            shadow = ShadowCalculation(fragPosLightSpace, bias);
        } else {
            shadow = 1.0;
        }
    }

    vec3 lighting = ambient + (diffuse + specular) * (1.0 - shadow);

    // Final colour: leaf over nearest-leaf background, lit by world-space normal.
    outColor.rgb = mix(bgAlbedo, leafAlbedo.rgb, weight) * lighting;
    outColor.a   = weight;
}

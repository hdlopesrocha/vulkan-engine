#version 450

// xy=UV, z=float(layerIdx) packed by the geometry shader.
layout(location = 0) in vec3 inTexCoord;
layout(location = 1) in vec3 inWorldPos;
layout(location = 2) flat in vec3 inFaceNormal;
layout(location = 3) flat in float inRotFrac;

layout(location = 0) out vec4 outColor;

#include "includes/ubo.glsl"

layout(set = 0, binding = 4) uniform sampler2D shadowMap;
layout(set = 0, binding = 8) uniform sampler2D shadowMap1;
layout(set = 0, binding = 9) uniform sampler2D shadowMap2;

// 60-layer impostor arrays: 3 billboard types × 20 Fibonacci views.
layout(set = 1, binding = 0) uniform sampler2DArray impostorArray;
layout(set = 1, binding = 1) uniform sampler2DArray impostorNormalArray;

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
    vec4 color = texture(impostorArray, inTexCoord);
    if (color.a < 0.5) discard;
    fragPosWorld = inWorldPos; // must be set before any ShadowCalculation call

    // Cross-fade with vegetation: dithered fade-in in the transition zone.
    // This is the complement of vegetation.frag's fade-out: together they cover
    // 100% of pixels so the transition is seamless without gaps or doubles.
    if (impostorDistance > 0.0) {
        float dist      = distance(ubo.viewPos.xyz, inWorldPos);
        float fadeAlpha = 1.0 - smoothstep(impostorDistance * 0.85, impostorDistance * 1.15, dist);
        const int M[16] = int[16](0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);
        float threshold = float(M[(int(gl_FragCoord.y) & 3) * 4 + (int(gl_FragCoord.x) & 3)]) / 16.0;
        if (threshold < fadeAlpha) discard;   // complementary: keep where vegetation discards
    }

    // Decode baked world-space normal from the normal capture array.
    // The capture was done for a canonical (theta=0) plant orientation, so we must
    // rotate the decoded normal by the per-instance Y-axis rotation to match the
    // actual plant orientation at runtime (same rotateY convention as vegetation_common.glsl).
    vec3 N_raw = normalize(texture(impostorNormalArray, inTexCoord).rgb * 2.0 - 1.0);
    float theta = inRotFrac * 6.28318530718;
    float cosT  = cos(theta);
    float sinT  = sin(theta);
    vec3 N = normalize(vec3(cosT * N_raw.x - sinT * N_raw.z,
                            N_raw.y,
                            sinT * N_raw.x + cosT * N_raw.z));
    vec3 L     = normalize(-ubo.lightDir.xyz);
    float NdotL = max(dot(N, L), 0.0);

    vec3  V     = normalize(ubo.viewPos.xyz - inWorldPos);
    vec3  H     = normalize(L + V);
    float NdotH = (NdotL > 0.0) ? max(dot(N, H), 0.0) : 0.0;

    const float kAmbient  = 0.30;
    const float kSpecular = 0.08;
    const float kShine    = 16.0;
    vec3 ambient  = kAmbient            * ubo.lightColor.rgb;
    vec3 diffuse  = NdotL               * ubo.lightColor.rgb;
    vec3 specular = pow(NdotH, kShine) * kSpecular * ubo.lightColor.rgb;

    float shadow = 0.0;
    if (ubo.shadowEffects.w > 0.5) {
        if (NdotL > 0.01) {
            float bias = max(0.003 * (1.0 - NdotL), 0.001);
            vec4 fragPosLightSpace = ubo.lightSpaceMatrix * vec4(inWorldPos, 1.0);
            shadow = ShadowCalculation(fragPosLightSpace, bias);
        } else {
            shadow = 1.0;
        }
    }

    vec3 lighting = ambient + (diffuse + specular) * (1.0 - shadow);

    outColor = vec4(color.rgb * lighting, 1.0);
}


#version 450

#include "includes/locations.glsl"

// xy=UV, z=float(layerIdx) packed by the geometry shader.
layout(location = VARY_UV) in vec3 inTexCoord;
layout(location = VARY_POSWORLD) in vec3 inWorldPos;
layout(location = VARY_FACE_NORMAL) flat in vec3 inFaceNormal;
layout(location = VARY_ROTFRAC) flat in float inRotFrac;
layout(location = VARY_POSLIGHT) flat in vec3 inInstanceOffset;

layout(location = FRAG_OUT_COLOR) out vec4 outColor;

#include "includes/ubo.glsl"
#include "includes/debug_modes.glsl"
#include "includes/sky_view.glsl"
#include "includes/perlin.glsl"
#include "includes/clouds.glsl"

layout(set = 0, binding = 4) uniform sampler2D shadowMap;
layout(set = 0, binding = 8) uniform sampler2D shadowMap1;
layout(set = 0, binding = 9) uniform sampler2D shadowMap2;

// 80-layer impostor arrays: 4 billboard types (3 vegetation + fire) × 20 Fibonacci views.
layout(set = 1, binding = 0) uniform sampler2DArray impostorArray;
layout(set = 1, binding = 1) uniform sampler2DArray impostorNormalArray;

// Depth data for deferred depth test: reconstruct world position from captured
// depth and write gl_FragDepth so the EQUAL compare in the shading pass matches
// the depth written by the impostor depth prepass.
layout(set = 1, binding = 2) uniform sampler2DArray depthArray;
layout(set = 1, binding = 3) readonly buffer CaptureInvVP {
    mat4 invVP[];
};

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

#include "includes/shadows.glsl"

void main() {
    // Debug: raw LOD source, no shading. Blue = impostor (this shader),
    // red = billboard (vegetation.frag). The color edge is the hand-off.
    if (ubo.debugMode == DEBUG_MODE_VEGETATION_LOD) {
        outColor = vec4(0.0, 0.0, 1.0, 1.0);
        return;
    }

    vec4 color = texture(impostorArray, inTexCoord);
    // Fire keeps its faint halo (same 0.02 floor the billboards and the
    // capture use) so apparent flame size matches; vegetation stays
    // alpha-tested at 0.3.
    int impBi = int(inTexCoord.z / 20.0);
    if (color.a < ((impBi == FIRE_BILLBOARD_INDEX) ? 0.02 : 0.3)) discard;
    fragPosWorld = inWorldPos; // must be set before any ShadowCalculation call

    // Direct hand-off (see impostors.vert): this instance only exists past
    // impostorDistance, where the billboard vertex shader has already culled
    // its counterpart. No dithered fade - the old cross-fade thinned the
    // grass to sparse dots instead of swapping to the impostor.

    // Reconstruct per-pixel depth from captured depth map so the deferred
    // depth test (EQUAL compare) shades only the nearest fragments.
    {
        int layer = int(inTexCoord.z);
        float texDepth = texture(depthArray, inTexCoord).r;
        if (texDepth >= 1.0 || texDepth <= 0.0) discard;

        vec2 ndc_xy = inTexCoord.xy * 2.0 - 1.0;
        vec4 clipPos = vec4(ndc_xy, texDepth, 1.0);
        vec4 worldPos = invVP[layer] * clipPos;
        worldPos /= worldPos.w;
        worldPos.xyz += inInstanceOffset;

        vec4 camClipPos = ubo.viewProjection * worldPos;
        gl_FragDepth = clamp(camClipPos.z / camClipPos.w, 0.0, 1.0);
    }

    // Fire glows: captured snapshot colors with captured coverage, no
    // lighting or shadow (flames are a light source, not a lit surface).
    // impBi was derived above for the alpha test.
    if (impBi == FIRE_BILLBOARD_INDEX) {
        // The snapshot holds single-plane coverage, but the billboards blend
        // several overlapping planes (alpha saturates toward 1 in the core).
        // Re-accumulate alpha approximately so distant flames match in
        // density. Snapshot rgb is already straight flame color: with the
        // opaque impostor pipeline it lands in the target unmultiplied, which
        // is exactly what the composite mixes by.
        float accA = 1.0 - pow(1.0 - color.a, 3.0);
        outColor = vec4(color.rgb, accA);
        return;
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
    vec3 L     = normalize(-ubo.lightDirection);
    float NdotL = max(dot(N, L), 0.0);

    vec3  V     = normalize(ubo.viewPosition - inWorldPos);
    vec3  H     = normalize(L + V);
    float NdotH = (NdotL > 0.0) ? max(dot(N, H), 0.0) : 0.0;

    const float kAmbient  = 0.30;
    const float kSpecular = 0.08;
    const float kShine    = 16.0;
    vec3 ambient  = kAmbient            * ubo.lightColor;
    vec3 diffuse  = NdotL               * ubo.lightColor;
    vec3 specular = pow(NdotH, kShine) * kSpecular * ubo.lightColor;

    float shadow = 0.0;
    if (ubo.shadowsEnabled) {
        if (NdotL > 0.01) {
            float bias = max(0.0005 * (1.0 - NdotL), 0.00005);
            vec4 fragPosLightSpace = ubo.lightSpaceMatrix * vec4(inWorldPos, 1.0);
            shadow = ShadowCalculationHard(fragPosLightSpace, inWorldPos, bias);
        } else {
            shadow = 1.0;
        }
    }
    if (sky.cloudsEnabled && NdotL > 0.01) {
        float cloudShadow = cloudShadowAt(inWorldPos);
        shadow = 1.0 - (1.0 - shadow) * (1.0 - cloudShadow);
    }

    vec3 lighting = ambient + (diffuse + specular) * (1.0 - shadow);

    outColor = vec4(color.rgb * lighting, 1.0);
}

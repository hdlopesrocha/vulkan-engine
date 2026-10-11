
#version 450

#include "../../includes/Locations.glsl"

// xy=UV, z=float(layerIdx) packed by the geometry shader.
layout(location = VARY_UV) in vec3 inTexCoord;
layout(location = VARY_POSWORLD) in vec3 inWorldPos;
// M5 (perf report 28): dead inFaceNormal declaration removed (sole producer
// was the free VS copy faceNorm = toCamera; no statement below reads it).
// H8 (perf report 27): flat per-instance trig from the vertex stage (same
// location as the old rotFrac it replaces) + flat normalized light direction.
layout(location = VARY_ROTFRAC) flat in vec2 inCosSinT;
layout(location = VARY_TANGENTWS) flat in vec3 inLightDirN;
layout(location = VARY_POSLIGHT) flat in vec3 inInstanceOffset;

layout(location = FRAG_OUT_COLOR) out vec4 outColor;

#include "../../includes/SceneBindings.glsl"
#include "../../includes/DebugModes.glsl"
#include "../../includes/sky/SkyView.glsl"
#include "../../includes/noise/Perlin.glsl"
#include "../../includes/sky/Clouds.glsl"

layout(set = 0, binding = 4) uniform sampler2D shadowMap;
layout(set = 0, binding = 8) uniform sampler2D shadowMap1;
layout(set = 0, binding = 9) uniform sampler2D shadowMap2;

// 60-layer impostor arrays: 3 vegetation types × 20 Fibonacci views.
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

#include "../../includes/shadow/Shadows.glsl"

void main() {
    // Debug: raw LOD source, no shading. Blue = impostor (this shader),
    // red = billboard (VegetationRenderer.frag). The color edge is the hand-off.
    if (ubo.debugMode == DEBUG_MODE_VEGETATION_LOD) {
        outColor = vec4(0.0, 0.0, 1.0, 1.0);
        return;
    }

    // M9 (perf report 27): depth tap before the albedo tap. The two
    // discard conditions (captured-depth range vs albedo alpha) are
    // independent, so the survivor set is order-free; depth-culled
    // background pixels skip the albedo fetch entirely. Survivors compute
    // identical values (both are array fetches at the same coord, no
    // derivatives involved).
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

    vec4 color = texture(impostorArray, inTexCoord);
    if (color.a < 0.3) discard;
    fragPosWorld = inWorldPos; // must be set before any ShadowCalculation call

    // Direct hand-off (see impostors.vert): this instance only exists past
    // impostorDistance, where the billboard vertex shader has already culled
    // its counterpart. No dithered fade - the old cross-fade thinned the
    // grass to sparse dots instead of swapping to the impostor.

    // Decode baked world-space normal from the normal capture array.
    // The capture was done for a canonical (theta=0) plant orientation, so we must
    // rotate the decoded normal by the per-instance Y-axis rotation to match the
    // actual plant orientation at runtime (same rotateY convention as vegetation_common.glsl).
    vec3 N_raw = normalize(texture(impostorNormalArray, inTexCoord).rgb * 2.0 - 1.0);
    // H8 (perf report 27): cosT/sinT arrive as flat varyings (same values the
    // per-pixel rebuild produced); the rotate/normalize/lighting chain below
    // runs the identical operations in the identical order.
    float cosT = inCosSinT.x;
    float sinT = inCosSinT.y;
    vec3 N = normalize(vec3(cosT * N_raw.x - sinT * N_raw.z,
                            N_raw.y,
                            sinT * N_raw.x + cosT * N_raw.z));
    // H8 (perf report 27): uniform light direction already normalized per
    // vertex (flat varying), not per pixel.
    vec3 L     = inLightDirN;
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
    if (ubo.shadowsEnabled != 0u) {
        if (NdotL > 0.01) {
            float bias = max(0.0005 * (1.0 - NdotL), 0.00005);
            vec4 fragPosLightSpace = ubo.lightSpaceMatrix * vec4(inWorldPos, 1.0);
            shadow = ShadowCalculationHard(fragPosLightSpace, inWorldPos, bias);
        } else {
            shadow = 1.0;
        }
    }
    if ((sky.cloudsEnabled != 0u) && NdotL > 0.01) {
        float cloudShadow = cloudShadowAt(inWorldPos);
        shadow = 1.0 - (1.0 - shadow) * (1.0 - cloudShadow);
    }

    vec3 lighting = ambient + (diffuse + specular) * (1.0 - shadow);

    outColor = vec4(color.rgb * lighting, 1.0);
}

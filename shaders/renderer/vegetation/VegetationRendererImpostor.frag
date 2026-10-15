
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
#include "../../includes/vegetation/GrassShading.glsl"

void main() {
    // Debug: raw LOD source, no shading. Blue = impostor (this shader),
    // red = blades (VegetationRenderer.frag). The color edge is the hand-off.
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
    //
    // The depth-reconstructed blade surface point drives shading (albedo
    // variation cells, shadow sampling, view vector) exactly like the blade
    // path's fragment world pos, so the two LODs agree.
    vec3 bladePos = inWorldPos;
    {
        int layer = int(inTexCoord.z);
        float texDepth = texture(depthArray, inTexCoord).r;
        if (texDepth >= 1.0 || texDepth <= 0.0) discard;

        vec2 ndc_xy = inTexCoord.xy * 2.0 - 1.0;
        vec4 clipPos = vec4(ndc_xy, texDepth, 1.0);
        vec4 worldPos = invVP[layer] * clipPos;
        worldPos /= worldPos.w;
        worldPos.xyz += inInstanceOffset;
        bladePos = worldPos.xyz;

        vec4 camClipPos = ubo.viewProjection * worldPos;
        gl_FragDepth = clamp(camClipPos.z / camClipPos.w, 0.0, 1.0);
    }

    vec4 color = texture(impostorArray, inTexCoord);
    if (color.a < 0.3) discard;

    // Direct hand-off (see impostors.vert): this instance only exists past
    // impostorDistance, where the blade vertex shader has already culled
    // its counterpart. No dithered fade - the old cross-fade thinned the
    // grass to sparse dots instead of swapping to the impostor.

    // Captured albedo has no per-tuft variation baked in; apply the shared
    // formula at the reconstructed pos so it matches the blade path exactly.
    vec3 albedo = color.rgb * (1.0 + grassVariation(bladePos));

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
    vec3 Ngeom = normalize(vec3(cosT * N_raw.x - sinT * N_raw.z,
                                N_raw.y,
                                sinT * N_raw.x + cosT * N_raw.z));
    vec3 N = Ngeom;
    // Two-sided, like the blade path (which flips by triangle winding):
    // keep the hemisphere facing the viewer so back-facing blades light up
    // instead of collapsing to ambient-only.
    vec3 Vv = normalize(ubo.viewPosition - bladePos);
    if (dot(N, Vv) < 0.0) N = -N;
    // H8 (perf report 27): uniform light direction already normalized per
    // vertex (flat varying), not per pixel.
    //
    // Canonical debug views 1/2/3 (see DebugModes.glsl): shading normal is
    // the two-sided N above, geometric is the captured normal before the
    // viewer flip, face is the rasterizer facet of the reconstructed
    // blade surface.
    if (ubo.debugMode == DEBUG_MODE_SHADING_NORMAL) {
        outColor = vec4(normalize(N) * 0.5 + 0.5, 1.0);
        return;
    }
    if (ubo.debugMode == DEBUG_MODE_GEOMETRIC_NORMAL) {
        outColor = vec4(Ngeom * 0.5 + 0.5, 1.0);
        return;
    }
    if (ubo.debugMode == DEBUG_MODE_FACE_NORMAL) {
        vec3 fn = normalize(cross(dFdy(bladePos), dFdx(bladePos)));
        outColor = vec4(fn * 0.5 + 0.5, 1.0);
        return;
    }
    vec3 col = grassShade(albedo, N, inLightDirN, bladePos, false);

    outColor = vec4(col, 1.0);
}

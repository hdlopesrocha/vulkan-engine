
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
// xyz = instance base, w = runtime height scale (hs).
layout(location = VARY_POSLIGHT) flat in vec4 inInstanceOffset;
// Terrain surface normal (flat): captured normals are flat-land, so the
// fragment tilts them onto this exactly like the blade VS does.
layout(location = VARY_PLANE_NORMAL) flat in vec3 inSurfN;

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

    // Hardware mip level of this fragment. Computed here, before any
    // discard, so the derivatives are always taken in uniform control flow.
    vec2 uvDx = dFdx(inTexCoord.xy);
    vec2 uvDy = dFdy(inTexCoord.xy);
    float hwLod = log2(max(length(uvDx), length(uvDy)) * float(textureSize(impostorArray, 0).x));

    // Exact texel for all per-texel data (depth, normal). Filtering device-Z
    // at silhouettes mixes blade depth with the cleared background, and —
    // the real defect — filtering/averaging NORMALS mixes ANTI-PARALLEL
    // normals: capture blades are 4-8 px wide with sub-texel tips, so a 2x2
    // filter cancels front/back sides of the same blade and neighbors into a
    // near-zero residual that normalize() amplifies into a garbage direction
    // (then re-flipped by the viewer test). Those pixels lit dark and wrong
    // — the "impostor normals are wrong / impostors darker than blades"
    // report. One exact texel per fragment keeps every value mutually
    // consistent and saturated, exactly like the shadow caster.
    ivec2 texSize = textureSize(depthArray, 0).xy;
    ivec2 texel = clamp(ivec2(inTexCoord.xy * vec2(texSize)), ivec2(0), texSize - ivec2(1));

    // M9 (perf report 27): depth tap before the albedo tap. The two
    // discard conditions (captured-depth range vs albedo alpha) are
    // independent, so the survivor set is order-free; depth-culled
    // background pixels skip the albedo fetch entirely.
    //
    // The depth-reconstructed blade surface point drives shading (albedo
    // variation cells, shadow sampling, view vector) exactly like the blade
    // path's fragment world pos, so the two LODs agree.
    vec3 bladePos = inWorldPos;
    {
        int layer = int(inTexCoord.z);
        float texDepth = texelFetch(depthArray, ivec3(texel, layer), 0).r;
        if (texDepth >= 1.0 || texDepth <= 0.0) discard;

        vec2 ndc_xy = inTexCoord.xy * 2.0 - 1.0;
        vec4 clipPos = vec4(ndc_xy, texDepth, 1.0);
        vec4 capPos = invVP[layer] * clipPos;
        capPos /= capPos.w;

        // Reconstruct the runtime blade surface: the captured point lives in
        // the canonical capture frame (origin, hs=1, no rotation), so the
        // exact runtime position is base + tilt_surfN(rotateY(hs * cap)) —
        // the same transform the blade VS bakes in and the placement scales
        // the quad by. Skipping hs/rotation/tilt skewed bladePos (and the
        // written depth) from the rasterized tuft by up to ~40% of its size.
        vec3 worldPos = inInstanceOffset.xyz + grassTiltToNormal(
            rotateY(capPos.xyz * inInstanceOffset.w, inCosSinT.x, inCosSinT.y),
            normalize(inSurfN));
        bladePos = worldPos;

        vec4 camClipPos = ubo.viewProjection * vec4(worldPos, 1.0);
        gl_FragDepth = clamp(camClipPos.z / camClipPos.w, 0.0, 1.0);
    }

    // LOD policy. Albedo: distance-ramped (a color averages correctly and
    // the averaged tone matches the blade canopy; also keeps far-field
    // coverage stable). All per-texel GEOMETRY data (depth, normal) is read
    // exactly, never filtered or mipped — see the texel comment above.
    float impostorDist = distance(ubo.viewPosition, bladePos);
    float lodCap = clamp(1.0 + 2.0 * log2(max(impostorDist, 700.0) / 700.0), 1.0, 8.0);
    float lod = min(hwLod, lodCap);

    vec4 color = textureLod(impostorArray, inTexCoord, lod);
    if (color.a < 0.3) discard;

    // Direct hand-off (see impostors.vert): this instance only exists past
    // impostorDistance, where the blade vertex shader has already culled
    // its counterpart. No dithered fade - the old cross-fade thinned the
    // grass to sparse dots instead of swapping to the impostor.

    // Captured albedo is premultiplied-form (opaque texels alpha 1,
    // background 0): divide the mipped alpha back out (no-op at L0), then
    // apply the shared variation formula so it matches the blade path.
    vec3 albedo = (color.rgb / max(color.a, 1e-3)) * (1.0 + grassVariation(bladePos));

    // Decode the baked world-space normal from the SAME exact capture texel
    // the depth came from (never filtered/mipped: averaging anti-parallel
    // blade normals cancels them into garbage — dark, wrongly-coloured
    // shading). The capture ran for a canonical (theta=0) plant, so rotate
    // by the per-instance Y-rotation (same rotateY convention as the blade
    // vertex shader), tilt onto the surface and flip toward the viewer.
    vec4 normalTex = texelFetch(impostorNormalArray, ivec3(texel, int(inTexCoord.z)), 0);
    vec3 N_raw = normalize(normalTex.rgb / max(normalTex.a, 1e-3) * 2.0 - 1.0);
    // H8 (perf report 27): cosT/sinT arrive as flat varyings (same values the
    // per-pixel rebuild produced); the rotate/normalize/lighting chain below
    // runs the identical operations in the identical order.
    float cosT = inCosSinT.x;
    float sinT = inCosSinT.y;
    vec3 Ngeom = grassTiltToNormal(normalize(vec3(cosT * N_raw.x - sinT * N_raw.z,
                                                       N_raw.y,
                                                       sinT * N_raw.x + cosT * N_raw.z)),
                                       normalize(inSurfN));
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

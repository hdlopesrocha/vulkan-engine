// Water TES (moved from water.tese). Requires: ubo, locations,
// perlin, water_noise. Defines the stage's main().
//
// C1 (perf report 19): this include also exposes the shared per-vertex wave
// core waterDisplaceWaterVertex() and the set-2 depth sampling helper
// waterShoreDirFromSamples() to the WATER_NO_TESS vertex path (main.vert built
// with -DWATER_NO_TESS=1). The no-tess VS measures fragWaterDepth/fragShoreDir
// with the exact same samples and sign rules as the TES, so the tessellated
// and non-tessellated geometry paths cannot drift. Only the TES
// layout/interface and the TES main() are compiled out under WATER_NO_TESS.

#ifndef WATER_NO_TESS

// Water tessellation evaluation shader
// Applies wave displacement using Perlin noise

layout(triangles, equal_spacing, cw) in;

layout(location = VARY_LOCALPOS) in vec3 inPos[];
layout(location = VARY_NORMAL) in vec3 inNormal[];
layout(location = VARY_SHARPNORMAL) in vec3 inBaseNormal[];
layout(location = VARY_UV) in vec2 inTexCoord[];
layout(location = VARY_BRUSHPATCH) in ivec3 tc_fragBrushIndex[];
layout(location = VARY_TEXWEIGHTS) in vec3 tc_fragTexWeights[];
layout(location = VARY_HSV) in vec3 tc_fragHSV[];

layout(location = VARY_LOCALPOS) out vec3 fragPos;
layout(location = VARY_NORMAL) out vec3 fragNormal;
layout(location = VARY_SHARPNORMAL) out vec3 fragBaseNormal;  // undisplaced base normal for per-fragment detail
layout(location = VARY_BASEPOS) out vec4 fragBasePos;        // xyz = undisplaced base position, w = raw bump amplitude
layout(location = VARY_WATERDEPTH) out float fragWaterDepth; // measured water depth (-1 = unknown/deep)
layout(location = VARY_SHOREDIR) out vec2 fragShoreDir;      // unit shore direction (toward thinner water)
layout(location = VARY_UV) out vec2 fragTexCoord;
layout(location = VARY_POSCLIP) out vec4 fragPosClip;  // clip-space position for depth lookup
layout(location = VARY_DEBUG) out vec3 fragDebug;   // debug visual (displacement)
layout(location = VARY_POSWORLD) out vec3 fragPosWorld;  // world-space position for shadow cascades
layout(location = VARY_POSLIGHT) out vec4 fragPosLightSpace; // light-space pos (cascade 0)
layout(location = VARY_BRUSHPATCH) flat out int fragBrushIndex;
layout(location = VARY_HSV) out vec3 fragHSV;

#endif // !WATER_NO_TESS

// Solid terrain depth (set 2, binding 5). The shore-wave zones are driven by
// the water depth measured against the REAL bottom here, not by the water
// volume's back face: the volume underside is pushed an SDF bias below the
// terrain (to keep the surface from z-fighting the ground), which shifts every
// zone by that bias. This binding is also used by the fragment stage, so the
// pass already depends on the solid depth target.
layout(set = 2, binding = 5) uniform sampler2D solidSceneDepthTex;
// Water volume back-face depth (set 2, binding 0): the second raster depth
// source. Where no solid bottom stands behind the pixel (deep/far water) the
// volume underside carries the water column; its constant SDF bias only
// shifts the absolute depth.
layout(set = 2, binding = 0) uniform sampler2D waterBackDepthTex;

#ifdef RT_ENABLED
// Inline ray-query water depth (optional, toggled by rt.waterDepth.x): trace
// the view ray to the exact solid bottom and take the world-space vertical
// drop. Same world-space definition as the raster path, so the shore-wave
// region zones are identical between modes.
#include "rt_params.glsl"
layout(set = 0, binding = 14) uniform accelerationStructureEXT rtTlas;
layout(set = 0, binding = 17) uniform RTBlock { RayTracingParamsGLSL rt; };
// Per-op RT profiling (RT_PROFILE variants only): set 0 binding 26 + macros.
#include "rt_profile.glsl"
#endif

// World-space direction of DECREASING water depth (toward the shore) from a
// center raw-depth sample plus two offset samples. Returns vec2(0) when the
// signal is unusable (clear depth, solid above the surface/in front, degenerate
// span, flat bottom).
vec2 waterShoreDirFromSamples(float rawC, float rawX, float rawY,
                              vec2 uvC, vec2 uvX, vec2 uvY, vec3 surfacePos) {
    if (rawC >= 1.0 || rawX >= 1.0 || rawY >= 1.0) return vec2(0.0);
    vec4 wC = ubo.invViewProjection * vec4(uvC * 2.0 - 1.0, rawC, 1.0);
    vec4 wX = ubo.invViewProjection * vec4(uvX * 2.0 - 1.0, rawX, 1.0);
    vec4 wY = ubo.invViewProjection * vec4(uvY * 2.0 - 1.0, rawY, 1.0);
    vec3 pC = wC.xyz / wC.w;
    vec3 pX = wX.xyz / wX.w;
    vec3 pY = wY.xyz / wY.w;
    float dC = surfacePos.y - pC.y;
    float dX = surfacePos.y - pX.y;
    float dY = surfacePos.y - pY.y;
    // A solid hit ABOVE the water surface is the terrain itself in front
    // (bank/cliff occluding the water), not a bottom: the gradient is
    // unusable there.
    if (dC < 0.0 || dX < 0.0 || dY < 0.0) return vec2(0.0);
    // Screen step -> world XZ offsets; solve the 2x2 system for the
    // world-space depth gradient, then take its negative (decreasing depth).
    vec2 sX = pX.xz - pC.xz;
    vec2 sY = pY.xz - pC.xz;
    float det = sX.x * sY.y - sX.y * sY.x;
    if (abs(det) < 1e-4) return vec2(0.0);
    vec2 g = vec2((dX - dC) * sY.y - (dY - dC) * sX.y,
                  sX.x * (dY - dC) - sY.x * (dX - dC)) / det;
    float gl = length(g);
    if (gl < 1e-5) return vec2(0.0);
    return -g / gl;
}

// ── Shared per-vertex wave core ─────────────────────────────────────────
// Displaces `pos` along its base normal by the thickness-zoned wave field and
// returns the analytic wave normal (exact for the `base + N * h` height
// field). The TES full path below and the WATER_NO_TESS vertex shader both
// call this, so the two geometry paths can never drift.
struct WaterVertexWave {
    vec3 pos;           // displaced world position
    vec3 normal;        // analytic wave normal at the displaced surface
    vec3 basePos;       // undisplaced base world position
    float displacement; // signed height displacement along the base normal
};

WaterVertexWave waterDisplaceWaterVertex(vec3 pos, vec3 normal, float animTime,
                                         float waterDepth, vec2 shoreDir,
                                         float bumpAmp, WaterParamsGPU wp) {
    // Surface basis (tangent plane) for projecting the analytic gradient.
    vec3 upVec = abs(normal.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(upVec, normal));
    vec3 B = cross(normal, T);

    vec4 wave = waterWaveSample(
        pos.xyz,
        animTime,
        waterDepth,
        bumpAmp,
        shoreDir,
        wp
    );

    // Project the analytic gradient onto the tangent basis to get the
    // height-field slopes along T and B.
    float dhdT = dot(wave.yzw, T);
    float dhdB = dot(wave.yzw, B);

    vec3 bumpedN = normalize(normal - dhdT * T - dhdB * B);
    if (dot(bumpedN, normal) < 0.0) bumpedN = -bumpedN;

    WaterVertexWave v;
    v.displacement = wave.x;
    // Displace along the FLAT base normal, not the perturbed one. The analytic
    // normal `bumpedN = N - dHdT*T - dHdB*B` is the exact surface normal only for
    // a height field defined as `base + N * h`. Displacing along the tilted
    // `bumpedN` instead would build a different surface whose true normal no
    // longer matches `bumpedN`, so lighting would disagree with the geometry
    // (visible especially on steep/large waves). Keeping the displacement axis
    // fixed at `normal` makes the rasterized surface and the shading normal
    // consistent.
    v.pos = pos + wave.x * normal;
    v.basePos = pos;
    v.normal = bumpedN;
    return v;
}

#ifndef WATER_NO_TESS

void main() {
    // Interpolate position
        vec3 bary = gl_TessCoord;

        // Interpolate position
        vec3 pos = bary.x * inPos[0] +
                   bary.y * inPos[1] +
                   bary.z * inPos[2];
    
    // Interpolate normal
        vec3 normal = normalize(bary.x * inNormal[0] +
                                bary.y * inNormal[1] +
                                bary.z * inNormal[2]);
    
    // Interpolate texture coordinates
        fragTexCoord = bary.x * inTexCoord[0] +
                       bary.y * inTexCoord[1] +
                       bary.z * inTexCoord[2];
    
    // Interpolate HSV
    fragHSV = tc_fragHSV[0] * bary.x + tc_fragHSV[1] * bary.y + tc_fragHSV[2] * bary.z;

    // Select per-patch brushIndex from compressed TCS outputs (tc_fragBrushIndex / tc_fragTexWeights)
    ivec3 texIndices = max(tc_fragBrushIndex[0], ivec3(0));
    vec3 weights = tc_fragTexWeights[0] * bary.x + tc_fragTexWeights[1] * bary.y + tc_fragTexWeights[2] * bary.z;
    int chosenIdx = texIndices.x;
    if (texIndices.y >= 0 && weights.y > weights.x) chosenIdx = texIndices.y;
    if (texIndices.z >= 0 && weights.z > max(weights.x, weights.y)) chosenIdx = texIndices.z;
    if (chosenIdx < 0) chosenIdx = 0;
    // Expose the chosen brushIndex to the fragment stage (kept raw so the
    // fragment debug view can show the true id distribution)
    fragBrushIndex = chosenIdx;

    // Load selected WaterParams from SSBO, falling back to layer 0 for
    // out-of-range terrain paint ids (see water.frag).
    int nWL = max(waterParams.length(), 1);
    WaterParamsGPU wp = waterParams[(chosenIdx >= 0 && chosenIdx < nWL) ? chosenIdx : 0];

    // ── C1 fast path: global tessellation toggle is OFF ────────────────
    // The TCS already clamped every tessellation level to 1, so this stage
    // runs once per base vertex. Emit the interpolated base vertex directly
    // and skip ALL texture fetches, the shore-gradient solve, the displaced-UV
    // refinement and the full wave field. The fragment stage derives its own
    // analytic wave normal from fragBasePos/fragWaterDepth/fragShoreDir, so
    // the base normal below is only a fallback.
    if (ubo.passParams.y < 0.5) {
        fragBaseNormal = normal;                      // undisplaced base normal
        fragBasePos = vec4(pos, wp.waveParams.z);     // base pos + raw bump amplitude
        // Shallow safe fallback depth (the TES-only depth textures are not
        // sampled here): finite waves, no deep swell, no contact foam.
        fragWaterDepth = clamp(wp.waveZones.z, 0.0, max(wp.waveZones.x, 1.0));
        fragShoreDir = normalize(wp.waveDirection.xy + vec2(1e-5, 0.0));
        fragNormal = normal;
        fragDebug = vec3(0.5);                        // zero displacement envelope
        fragPos = pos;
        fragPosWorld = pos;
        fragPosLightSpace = ubo.lightSpaceMatrix * vec4(pos, 1.0);
        vec4 fastClipPos = ubo.viewProjection * vec4(pos, 1.0);
        fragPosClip = fastClipPos;
        gl_Position = fastClipPos;
        return;
    }

    // Get the water parameters driving the single wave field.
    float time = waterRenderUBO.timeParams.x;
    float noiseTimeSpeed = wp.params3.x;

    float bumpAmp = wp.waveParams.z; // bump amplitude provided via Water widget

    // --- Screen-space UV of the undisplaced base vertex ---
    // Used to sample the water back-face depth (thickness) below.
    vec2 screenUV = vec2(0.0);
    bool haveScreen = false;
    {
        vec4 preClip = ubo.viewProjection * vec4(pos, 1.0);
        if (preClip.w > 0.001) {
            screenUV = clamp(preClip.xy / preClip.w * 0.5 + 0.5, 0.001, 0.999);
            haveScreen = true;
        }
    }

    // --- Water depth + shore direction (drive the shore-wave zones) ---
    // The region zones read a VERTICAL world-space depth: the drop from the
    // water surface to the bottom, surfaceY - bottomY. Two interchangeable
    // sources:
    //   * Raster (always available): the solid scene depth (the visible
    //     terrain bottom) plus the water volume back-face depth (the volume
    //     underside), both reconstructed to world space. The shallowest valid
    //     drop wins, so the SDF-biased volume underside cannot deepen the
    //     zones and a missing solid falls back to the volume.
    //   * RT (rt.waterDepth.x, needs a built TLAS): the same view ray is
    //     traced to the exact solid bottom and the same vertical drop is
    //     computed, so the zone definition is identical between modes. On a
    //     miss the raster value stands.
    // There is deliberately no "unknown depth" state: the wave field maps a
    // negative depth to the full deep-ocean swell, whose hard boundary
    // against measured water printed as a seam in the surface normal/foam.
    //
    // The shore direction is the direction of DECREASING water depth (the
    // negative water-depth gradient): waves and their foam travel toward
    // thinning water. The gradient is a central sample of the solid depth at
    // two screen offsets, reconstructed in world space. In deep water the
    // solid bottom can be too far/absent, so the water volume's own back face
    // (whose constant SDF bias is vertical and does not change the horizontal
    // gradient direction) provides the direction fallback. Finally the
    // configured shoreWaveAngle is used when neither measures a slope.
    vec2 shoreDir = normalize(wp.waveDirection.xy + vec2(1e-5, 0.0));
    // Raster drop candidates (-1 = not measurable: clear depth, or a solid
    // sample above the water = terrain in front of the water).
    float solidDrop = -1.0;
    float backDrop = -1.0;
    float solidDepthRaw = 1.0;
    if (haveScreen) {
        solidDepthRaw = texture(solidSceneDepthTex, screenUV).r;
        if (solidDepthRaw < 1.0) {
            vec4 solidWorldH = ubo.invViewProjection * vec4(screenUV * 2.0 - 1.0, solidDepthRaw, 1.0);
            float drop = pos.y - solidWorldH.y / solidWorldH.w;
            solidDrop = (drop >= 0.0) ? drop : -1.0;
        }
        float backDepthRaw = texture(waterBackDepthTex, screenUV).r;
        if (backDepthRaw < 1.0) {
            vec4 backWorldH = ubo.invViewProjection * vec4(screenUV * 2.0 - 1.0, backDepthRaw, 1.0);
            backDrop = max(pos.y - backWorldH.y / backWorldH.w, 0.0);
        }
    }
    // Deepest credible bottom of the two world-space drops: the solid terrain
    // (the visible bed) and the water volume underside (pushed an SDF bias
    // below the terrain). Taking the deeper of the two keeps the depth
    // continuous where the solid sample is a bank/edge above the water
    // (negative drop, ignored) or disappears at range — the volume depth
    // covers it. -1 candidates (unmeasurable) lose to the valid one.
    float waterDepth = max(solidDrop, backDrop);
    bool waterDepthFromRt = false;
#ifdef RT_ENABLED
    if (rt.waterDepth.x > 0.5 && rt.debug.y > 0.5) {
        // Exact solid bottom along the same view ray the raster sample uses.
        vec3 rayD = normalize(pos - ubo.viewPos.xyz);
        RT_PROF_BEGIN(rtProfDepth, RT_PROFILE_OP_WATER_DEPTH);
        rayQueryEXT rq;
        rayQueryInitializeEXT(rq, rtTlas, gl_RayFlagsOpaqueEXT, RT_RAY_MASK_SCENE,
                              pos, 0.05, rayD, RT_NO_LIMIT);
        while (rayQueryProceedEXT(rq)) {}
        RT_PROF_END(rtProfDepth);
        if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT &&
            rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true) == RT_SCENE_INSTANCE) {
            RT_PROF_HIT(RT_PROFILE_OP_WATER_DEPTH);
            vec3 hitPos = pos + rayD * rayQueryGetIntersectionTEXT(rq, true);
            float drop = pos.y - hitPos.y;
            if (drop >= 0.0) {
                waterDepth = drop;
                waterDepthFromRt = true;
            }
        }
    }
#endif
    if (waterDepth < 0.0) {
        // Guard only (base vertex behind the camera, or no measurable raster
        // depth and an RT miss): stay finite so the wave field never enters
        // its unknown -> full-deep-swell default. Shallow zone boundary:
        // finite waves, no deep swell, no contact foam.
        waterDepth = clamp(wp.waveZones.z, 0.0, max(wp.waveZones.x, 1.0));
    }

    if (haveScreen) {
        float gradStep = max(wp.waveWarp.w, 0.0);
        if (gradStep > 0.0) {
            vec2 texel = 1.0 / vec2(textureSize(solidSceneDepthTex, 0));
            vec2 uvX = clamp(screenUV + vec2(texel.x * gradStep, 0.0), 0.0, 1.0);
            vec2 uvY = clamp(screenUV + vec2(0.0, texel.y * gradStep), 0.0, 1.0);

            vec2 dir = waterShoreDirFromSamples(
                solidDepthRaw,
                texture(solidSceneDepthTex, uvX).r,
                texture(solidSceneDepthTex, uvY).r,
                screenUV, uvX, uvY, pos);
            if (dot(dir, dir) < 1e-6) {
                dir = waterShoreDirFromSamples(
                    texture(waterBackDepthTex, screenUV).r,
                    texture(waterBackDepthTex, uvX).r,
                    texture(waterBackDepthTex, uvY).r,
                    screenUV, uvX, uvY, pos);
            }
            if (dot(dir, dir) > 1e-6) shoreDir = dir;
        }
    }
    fragShoreDir = shoreDir;

    // Calculate the thickness-zoned shore-wave displacement and its analytic
    // spatial gradient (single wave field: directional swell + chop + mask).
    // Shared with the WATER_NO_TESS vertex path via waterDisplaceWaterVertex()
    // so both geometry paths cannot drift.
    float animTime = time * noiseTimeSpeed;
    WaterVertexWave wv = waterDisplaceWaterVertex(pos, normal, animTime,
                                                  waterDepth, shoreDir, bumpAmp, wp);
    float waveDisplacement = wv.displacement;
    pos = wv.pos;
    fragNormal = wv.normal;
    fragBaseNormal = normal;   // undisplaced (flat) interpolated base normal
    // Recompute the base position from the displaced one (not wv.basePos) to
    // keep the tessellated path's floating-point sequence unchanged.
    fragBasePos = vec4(pos - waveDisplacement * normal, bumpAmp);  // base pos + raw amplitude

    // Refine the raster depth at the DISPLACED screen position. The depth
    // textures are sampled per pixel, but the depth above was read at the
    // BASE vertex's UV; at distance a large wave displacement against a
    // grazing view projects the base vertex tens of pixels from the shaded
    // fragment, so the base-UV sample can land on shore/sky texels and cut
    // the raster depth off. The displaced UV is the pixel actually shaded.
    // The RT path is a world-space ray and does not depend on the UV, so a
    // successful RT hit is never overwritten.
    if (!waterDepthFromRt) {
        vec4 dispClip = ubo.viewProjection * vec4(pos, 1.0);
        if (dispClip.w > 0.001) {
            vec2 uvD = clamp(dispClip.xy / dispClip.w * 0.5 + 0.5, 0.001, 0.999);
            float sd = texture(solidSceneDepthTex, uvD).r;
            float bd = texture(waterBackDepthTex, uvD).r;
            float sDrop = -1.0;
            float bDrop = -1.0;
            if (sd < 1.0) {
                vec4 w = ubo.invViewProjection * vec4(uvD * 2.0 - 1.0, sd, 1.0);
                float drop = pos.y - w.y / w.w;
                sDrop = (drop >= 0.0) ? drop : -1.0;
            }
            if (bd < 1.0) {
                vec4 w = ubo.invViewProjection * vec4(uvD * 2.0 - 1.0, bd, 1.0);
                bDrop = max(pos.y - w.y / w.w, 0.0);
            }
            float refined = max(sDrop, bDrop);
            if (refined >= 0.0) waterDepth = refined;
        }
    }
    fragWaterDepth = waterDepth;

    // Debug: encode displacement as color (normalized against the largest
    // possible envelope: deep + shoal gain + breaker bump).
    float maxExpected = max(bumpAmp * (1.0 + wp.waveShape.w + wp.waveBreaker.x), 1e-3);
    float normDisp = clamp((waveDisplacement / maxExpected) * 0.5 + 0.5, 0.0, 1.0);
    fragDebug = vec3(normDisp);
    
    fragPos = pos;
    fragPosWorld = pos;
    fragPosLightSpace = ubo.lightSpaceMatrix * vec4(pos, 1.0);
    vec4 clipPos = ubo.viewProjection * vec4(pos, 1.0);
    fragPosClip = clipPos;
    gl_Position = clipPos;
}

#endif // !WATER_NO_TESS

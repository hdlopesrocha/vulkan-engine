#version 450
#extension GL_ARB_shader_draw_parameters : require

#include "includes/ubo.glsl"
#include "includes/locations.glsl"

// Stage dispatcher (Phase-1 merge): WATER_MODE selects the water vertex path.
// WATER_NO_TESS (C1, perf report 19) selects the non-tessellation water path:
// the pipeline is TRIANGLE_LIST with no TCS/TES, so this vertex shader writes
// the water FRAGMENT interface directly (the same varyings the TES emits) and
// shares the per-vertex wave core in water_tese.glsl.
#ifndef WATER_MODE
#define WATER_MODE 0
#endif

layout(location = ATTR_POS) in vec3 inPos;
layout(location = ATTR_COLOR) in vec3 inColor;
layout(location = ATTR_UV) in vec2 inUV;
layout(location = ATTR_NORMAL) in vec3 inNormal;
layout(location = ATTR_BRUSH_INDEX) in int inBrushIndex;
layout(location = ATTR_HSV) in vec3 inHSV;

#if WATER_MODE && defined(WATER_NO_TESS)
// ── Non-tessellation water VS (shaders/main_water_no_tess.vert.spv) ────────
// Direct VS -> FS interface: the water fragment shader (and the back-face
// fragment shader, which declares the same varyings) consumes these
// locations. The no-tess path measures fragWaterDepth/fragShoreDir from the
// set-2 depth textures with the exact same samples, sign rules and
// displaced-UV refinement as the TES (water_tese.glsl), so the two paths
// render the same shore zones and region tint.
#include "includes/perlin.glsl"
#include "includes/water_noise.glsl"
#include "includes/water_tese.glsl"

layout(location = VARY_LOCALPOS) out vec3 fragPos;
layout(location = VARY_NORMAL) out vec3 fragNormal;
layout(location = VARY_SHARPNORMAL) out vec3 fragBaseNormal;  // undisplaced base normal for per-fragment detail
layout(location = VARY_BASEPOS) out vec4 fragBasePos;         // xyz = undisplaced base position, w = raw bump amplitude
layout(location = VARY_WATERDEPTH) out float fragWaterDepth;  // measured water depth (-1 = unknown/deep)
layout(location = VARY_SHOREDIR) out vec2 fragShoreDir;       // unit shore direction (toward thinner water)
layout(location = VARY_UV) out vec2 fragTexCoord;
layout(location = VARY_POSCLIP) out vec4 fragPosClip;         // clip-space position for depth lookup
layout(location = VARY_DEBUG) out vec3 fragDebug;             // debug visual (displacement)
layout(location = VARY_POSWORLD) out vec3 fragPosWorld;       // world-space position for shadow cascades
layout(location = VARY_POSLIGHT) out vec4 fragPosLightSpace;  // light-space pos (cascade 0)
layout(location = VARY_BRUSHPATCH) flat out int fragBrushIndex;
layout(location = VARY_HSV) out vec3 fragHSV;

void main() {
    // Identity model (models removed): inPos is already world space.
    vec3 pos = inPos;
    vec3 normal = normalize(inNormal);

    // Layer params by the vertex's own brush index. Without a TCS there is no
    // patch compression, so the vertex index is used directly; out-of-range
    // paint ids fall back to layer 0 exactly like the TES SSBO lookup.
    int chosenIdx = inBrushIndex;
    if (chosenIdx < 0) chosenIdx = 0;
    int nWL = max(waterParams.length(), 1);
    WaterParamsGPU wp = waterParams[(chosenIdx >= 0 && chosenIdx < nWL) ? chosenIdx : 0];

    float bumpAmp = wp.waveParams.z; // bump amplitude provided via Water widget

    // --- Base screen UV of the undisplaced vertex (same as the TES) ---
    vec2 screenUV = vec2(0.0);
    bool haveScreen = false;
    {
        vec4 preClip = ubo.viewProjection * vec4(pos, 1.0);
        if (preClip.w > 0.001) {
            screenUV = clamp(preClip.xy / preClip.w * 0.5 + 0.5, 0.001, 0.999);
            haveScreen = true;
        }
    }

    // --- Water depth + shore direction (identical set-2 samples and sign
    //     rules as the TES; the no-tess path has no RT depth branch, so this
    //     is the measured raster depth only) ---
    vec2 shoreDir = normalize(wp.waveDirection.xy + vec2(1e-5, 0.0));
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
    float waterDepth = max(solidDrop, backDrop);
    if (waterDepth < 0.0) {
        // Same guard as the TES: stay finite so the wave field never enters
        // its unknown -> full-deep-swell default (shallow zone boundary).
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

    // Per-vertex wave displacement + analytic normal via the SHARED core.
    float animTime = waterRenderUBO.timeParams.x * wp.params3.x;
    WaterVertexWave wv = waterDisplaceWaterVertex(pos, normal, animTime,
                                                  waterDepth, shoreDir, bumpAmp, wp);

    // Refine the measured depth at the DISPLACED vertex, exactly like the TES
    // (water_tese.glsl): the depth above was read at the base vertex's UV,
    // which at a grazing view can land far from the shaded fragment. There is
    // no RT depth path in the no-tess VS, so this always runs.
    {
        vec4 dispClip = ubo.viewProjection * vec4(wv.pos, 1.0);
        if (dispClip.w > 0.001) {
            vec2 uvD = clamp(dispClip.xy / dispClip.w * 0.5 + 0.5, 0.001, 0.999);
            float sd = texture(solidSceneDepthTex, uvD).r;
            float bd = texture(waterBackDepthTex, uvD).r;
            float sDrop = -1.0;
            float bDrop = -1.0;
            if (sd < 1.0) {
                vec4 w = ubo.invViewProjection * vec4(uvD * 2.0 - 1.0, sd, 1.0);
                float drop = wv.pos.y - w.y / w.w;
                sDrop = (drop >= 0.0) ? drop : -1.0;
            }
            if (bd < 1.0) {
                vec4 w = ubo.invViewProjection * vec4(uvD * 2.0 - 1.0, bd, 1.0);
                bDrop = max(wv.pos.y - w.y / w.w, 0.0);
            }
            float refined = max(sDrop, bDrop);
            if (refined >= 0.0) waterDepth = refined;
        }
    }

    fragBrushIndex = chosenIdx;
    fragHSV = inHSV;
    fragTexCoord = inUV;
    fragBaseNormal = normal;                  // undisplaced (flat) base normal
    fragBasePos = vec4(wv.basePos, bumpAmp);  // base pos + raw amplitude
    fragWaterDepth = waterDepth;
    fragShoreDir = shoreDir;
    fragNormal = wv.normal;
    fragDebug = vec3(0.5);                    // zero displacement envelope
    fragPos = wv.pos;
    fragPosWorld = wv.pos;
    fragPosLightSpace = ubo.lightSpaceMatrix * vec4(wv.pos, 1.0);
    vec4 clipPos = ubo.viewProjection * vec4(wv.pos, 1.0);
    fragPosClip = clipPos;
    gl_Position = clipPos;
}
#else
layout(location = VARY_COLOR) out vec3 fragColor;
layout(location = VARY_UV) out vec2 fragUV;
layout(location = VARY_NORMAL) out vec3 fragNormal;
layout(location = VARY_POSWORLD) out vec3 fragPosWorld;
layout(location = VARY_BRUSHPATCH) out int fragBrushIndex;      // per-vertex texture index for TCS
layout(location = VARY_POSLIGHT) out vec4 fragPosLightSpace;
layout(location = VARY_LOCALPOS) out vec3 fragLocalPos;          // provide local/world pos to TCS
layout(location = VARY_LOCALNORMAL) out vec3 fragLocalNormal;       // provide local/world normal to TCS
layout(location = VARY_SHARPNORMAL) out vec3 fragSharpNormal;      // face normal
layout(location = VARY_POSCLIP) out vec4 fragPosClip;              // clip-space pos (water back-face pass)
layout(location = VARY_HSV) out vec3 fragHSV;

void main() {
    fragColor = inColor;
    fragUV = inUV;
    // Models removed: always use identity model matrix
    mat4 model = mat4(1.0);
    // Transform normal to world space (model is identity here)
    // For uniform scaling, mat3(model) works. For non-uniform scaling, use transpose(inverse(model))
    fragNormal = normalize(mat3(model) * inNormal);
    
    // Pass per-vertex texture index as flat int for patch compression in TCS
    fragBrushIndex = inBrushIndex;
    
    // compute world-space position and pass to fragment
    vec4 worldPos = model * vec4(inPos, 1.0);
    fragPosWorld = worldPos.xyz;
    fragLocalPos = worldPos.xyz;       // Use world-space position as local basis for displacement
    
    // compute light-space position for shadow mapping
    fragPosLightSpace = ubo.lightSpaceMatrix * worldPos;
    
    // Compute face normal (sharp normal) from model normal
    fragLocalNormal = fragNormal;      // Propagate normal for tessellation stages
    fragSharpNormal = normalize(mat3(model) * inNormal);
    
    // Pass through per-vertex HSV
    fragHSV = inHSV;
    
    // apply MVP transform to the vertex position
    gl_Position = ubo.viewProjection * worldPos;
    fragPosClip = gl_Position;
}
#endif

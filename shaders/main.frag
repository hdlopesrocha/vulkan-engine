
#version 460
// RT_ENABLED is defined for the main_rt.frag.spv variant (hardware RT path).
// Without it the shader uses the procedural-sky fallback with identical miss
// baselines, so non-RT hardware stays validation-clean (no TLAS/ray queries).
#ifdef RT_ENABLED
#extension GL_EXT_ray_query : require
#endif
#include "includes/locations.glsl"

// Stage dispatcher (Phase-1 merge): WATER_MODE selects the water stage path.
// COMPILE-time: the water pipeline builds this source with -DWATER_MODE=1
// (shaders/main_water*.spv) and supplies the water varying/binding interface;
// terrain builds the default 0 (solid interface). One source, two variants —
// no spec constants, so the terrain pipelines never see the water code.
#ifndef WATER_MODE
#define WATER_MODE 0
#endif

#if !WATER_MODE
// Solid terrain fragment interface (water declares its own set in
// includes/water_frag_stage.glsl).
layout(location = VARY_COLOR) in vec3 fragColor;
layout(location = VARY_UV) in vec2 fragUV;
layout(location = VARY_NORMAL) in vec3 fragNormal;
layout(location = VARY_POSWORLD) in vec3 fragPosWorld;
layout(location = VARY_BRUSHPATCH) flat in ivec3 fragTexIndices;
layout(location = VARY_POSLIGHT) in vec4 fragPosLightSpace;
layout(location = VARY_LOCALPOS) in vec3 fragPosWorldNotDisplaced;
layout(location = VARY_TEXWEIGHTS) in vec3 fragTexWeights;
layout(location = VARY_HSV) in vec3 fragHSV;
layout(location = VARY_SHARPNORMAL) in vec3 fragSharpNormal; // face normal computed in TES (sharp)
layout(location = VARY_DEBUG) in vec3 fragTessLevel; // tessellation level heat (tesc output /16, DEBUG_MODE_TESS_HEAT)
#endif

#include "includes/ubo.glsl"

#include "includes/debug_modes.glsl"

#include "includes/textures.glsl"

#if !WATER_MODE && !defined(BRUSH_PASS)
layout(set = 1, binding = 0) uniform sampler2D brushDepthTex;
layout(set = 1, binding = 1) uniform sampler2D brushBackFaceDepthTex;
#endif

layout(location = FRAG_OUT_COLOR) out vec4 outColor;

#include "includes/common.glsl"

#include "includes/tbn.glsl"
#include "includes/triplanar.glsl"

#ifndef BRUSH_PASS
#include "includes/shadows.glsl"
#endif

#include "includes/hsv.glsl"

// Hybrid RT declarations (bindings 14/17/18 — TLAS, params, proxy metadata).
// The TLAS holds the stable solid-proxy boxes; water surface is excluded by
// design (origins, never targets). Shaders gate all sampling on rt.debug.y
// (tlasReady): before the first BLAS/TLAS build completes everything falls
// back to sky/CSM so no invalid acceleration structure is ever traced.
// rt_params.glsl is included UNCONDITIONALLY (structs + rtProceduralSky): the
// sky-reflection fallback used with RT off needs the procedural sky helper;
// the RT UBO/TLAS/buffer bindings stay compiled out without RT_ENABLED.
#include "includes/rt_params.glsl"
#ifdef RT_ENABLED
layout(set = 0, binding = 14) uniform accelerationStructureEXT rtTlas;
layout(set = 0, binding = 17) uniform RTBlock { RayTracingParamsGLSL rt; };
layout(set = 0, binding = 18) readonly buffer RTMeta { RTProxyMetaGLSL rtMetas[]; };
// Real scene-geometry lookups. The scene is split across two TLAS instances
// by content — solids (instance RT_SCENE_INSTANCE) and the real water mesh
// (instance RT_SCENE_WATER_INSTANCE) — each backed by its own BLAS. These
// combined buffers store the solids partition first and the water mesh
// second, while each BLAS numbers its geometries from 0:
//   rtScenePrimBase[0] = solid geometry count (water partition base, written
//                        by RayTracingResources::recordSceneBlas)
//   rtScenePrimBase[i] = first primitive of combined geometry i (unused by
//                        ray-query shading: primitive indices are per-BLAS)
//   rtSceneAlbedo[i]   = chunk average albedo / water layer
//   rtSceneGeomInfo[i] = {baseVertex, firstIndex, primBase, waterFlag}
// Any hit's PER-BLAS geometry index must be mapped through rtSceneGeomIndex()
// before indexing them (water indices are offset by the partition base).
layout(set = 0, binding = 21) readonly buffer RTScenePrimBase { uint rtScenePrimBase[]; };
layout(set = 0, binding = 22) readonly buffer RTSceneAlbedo { vec4 rtSceneAlbedo[]; };
// Real triangle attributes for hit shading: geometry bases, the merged vertex
// pool (Vertex = 16 floats: position 0-2, normal 8-10) and the index pool.
layout(set = 0, binding = 23) readonly buffer RTSceneGeomInfo { uvec4 rtSceneGeomInfo[]; };
layout(set = 0, binding = 24) readonly buffer RTSceneVerts { float rtSceneVerts[]; };
layout(set = 0, binding = 25) readonly buffer RTSceneIndices { uint rtSceneIndices[]; };

// Scene-instance tests and the per-BLAS -> combined-lookup index mapping.
// Every geometry read (geomInfo/albedo) goes through rtSceneGeomIndex().
bool rtIsSceneInstance(uint instanceCustomIndex) {
    return instanceCustomIndex == RT_SCENE_INSTANCE
        || instanceCustomIndex == RT_SCENE_WATER_INSTANCE;
}
uint rtSceneGeomIndex(uint instanceCustomIndex, uint geometryIndex) {
    return (instanceCustomIndex == RT_SCENE_WATER_INSTANCE)
        ? rtScenePrimBase[0] + geometryIndex : geometryIndex;
}
#include "includes/rt_scene_sample.glsl"
#include "includes/rt_reflection.glsl"
#endif

// Screen-space reflection refinement (set 0, bindings 19/20): the *previous*
// frame's solid HDR color/depth. The proxy ray query gives plausible but
// blocky reflections; where the reflected point is on screen, marching the
// real depth buffer and sampling the real color resolves a precise mirror.
layout(set = 0, binding = 19) uniform sampler2D ssrColorTex;
layout(set = 0, binding = 20) uniform sampler2D ssrDepthTex;


// Global toggles
bool roughnessEnabled = ubo.debugParams.y > 0.5;
bool aoEnabled = ubo.debugParams.z > 0.5;

#if WATER_MODE
// Water fragment stage (varyings + set-2 scene textures + shadeWaterSurface
// + main). Same source as water's old dedicated shader, merged into this
// file; see includes/water_frag_stage.glsl.
#include "includes/perlin.glsl"
#include "includes/water_noise.glsl"
#include "includes/water_frag_stage.glsl"
#else
// Solid surface shading (extracted): writes outColor; debug branches return.
#include "includes/solid_surface.glsl"
#endif

void main() {
#if WATER_MODE
    // Water surface shading writes outColor; alpha drives the blend in the
    // water pipeline (alpha-blended into the main color target).
    shadeWaterSurface();
#else
    bool isShadowPass = ubo.passParams.x > 0.5;
    // Provide a default color so early debug/special-case returns still
    // produce a valid output for downstream passes.
    outColor = vec4(0.0);

    // Fast-path for shadow pass: skip expensive lighting/texture work.
    if (isShadowPass) {
        outColor = vec4(0.0);
        return;
    }

    // Solid surface shading lives in includes/solid_surface.glsl;
    // it writes outColor (debug branches return early inside it).
    shadeSolidSurface();
#endif
}

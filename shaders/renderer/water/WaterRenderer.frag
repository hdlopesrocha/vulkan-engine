
#version 460
// RT_ENABLED is defined for the WaterRendererRT.frag.spv / WaterRendererRTNoBody.frag.spv /
// WaterRendererRTProf.frag.spv variants (hardware RT path). Without it the shader
// uses the procedural-sky fallback with identical miss baselines, so non-RT
// hardware stays validation-clean (no TLAS/ray queries).
#ifdef RT_ENABLED
#extension GL_EXT_ray_query : require
#endif
// Shader clock for per-op RT profiling (RT_PROFILE variants only). Must be a
// top-level #extension: directives inside includes are illegal after tokens.
#ifdef RT_PROFILE
#extension GL_EXT_shader_realtime_clock : require

#include "../../ssbo/RTProxyMetaGLSL.glsl"
#include "../../ubo/RayTracingParams.glsl"
#endif
#include "../../includes/Locations.glsl"

// ── Water surface fragment shader ─────────────────────────────────────────
// Split out of the old shared main.frag (which selected the path with
// -DWATER_MODE). Compile-time variants:
//   * default                    -> shaders/WaterRenderer.frag.spv
//   * -DRT_ENABLED               -> shaders/WaterRendererRT.frag.spv
//   * -DWATER_NO_BODY            -> shaders/WaterRendererNoBody.frag.spv
//   * -DRT_ENABLED -DWATER_NO_BODY -> shaders/WaterRendererRTNoBody.frag.spv
//   * -DRT_ENABLED -DRT_PROFILE  -> shaders/WaterRendererRTProf.frag.spv
#include "../../includes/SceneBindings.glsl"

// Depth-region water tint ramp: shared by the raster water surface and the
// water hit-shading inside RT rays (both include this helper).
#include "../../includes/water/WaterTint.glsl"

#include "../../includes/DebugModes.glsl"

#include "../../includes/Textures.glsl"

layout(location = FRAG_OUT_COLOR) out vec4 outColor;

// Water aux outputs (color attachments 1 and 2 of the water geometry pass):
//  body   = refraction + tint body (RGB) and body weight (A: coverage times
//           the inverse reflection mix), the only part the final composite
//           blurs — reflections and surface effects stay sharp.
//  column = measured water depth in meters, the blur radius driver.
// WATER_NO_BODY (H4): the single-color-attachment water variant used when no
// layer needs the final-pass blur. The outputs (and their writes in
// water_surface.glsl) are compiled out; all shading computation stays.
#ifndef WATER_NO_BODY
layout(location = FRAG_OUT_WATER_BODY) out vec4 outWaterBody;
layout(location = FRAG_OUT_WATER_COLUMN) out vec4 outWaterColumn;
#endif

#include "../../includes/Common.glsl"

#include "../../includes/solid/TangentBasis.glsl"
#include "../../includes/solid/Triplanar.glsl"

#include "../../includes/shadow/Shadows.glsl"

#include "../../includes/Hsv.glsl"

// Hybrid RT declarations (bindings 14/17/18 — TLAS, params, proxy metadata).
// See SolidRenderer.frag for the full scene-buffer notes; the water shading uses the
// same helpers.
#include "../../includes/rt/RtParams.glsl"
#ifdef RT_ENABLED
layout(set = 0, binding = 14) uniform accelerationStructureEXT rtTlas;
layout(std140, set = 0, binding = 17) uniform RTBlock { RayTracingParams rt; };
layout(set = 0, binding = 18) readonly buffer RTMeta { RTProxyMetaGLSL rtMetas[]; };
layout(set = 0, binding = 21) readonly buffer RTScenePrimBase { uint rtScenePrimBase[]; };
layout(set = 0, binding = 22) readonly buffer RTSceneAlbedo { vec4 rtSceneAlbedo[]; };
layout(set = 0, binding = 23) readonly buffer RTSceneGeomInfo { uvec4 rtSceneGeomInfo[]; };
layout(set = 0, binding = 24) readonly buffer RTSceneVerts { float rtSceneVerts[]; };
layout(set = 0, binding = 25) readonly buffer RTSceneIndices { uint rtSceneIndices[]; };

// Scene-instance tests and the per-BLAS -> combined-lookup index mapping.
bool rtIsSceneInstance(uint instanceCustomIndex) {
    return instanceCustomIndex == RT_SCENE_INSTANCE
        || instanceCustomIndex == RT_SCENE_WATER_INSTANCE;
}
uint rtSceneGeomIndex(uint instanceCustomIndex, uint geometryIndex) {
    return (instanceCustomIndex == RT_SCENE_WATER_INSTANCE)
        ? rtScenePrimBase[0] + geometryIndex : geometryIndex;
}
// Per-op RT profiling (RT_PROFILE variants only): declares set 0 binding 26
// and the RT_PROF_* macros used by the instrumented ray-query sites. Included
// BEFORE the reflection helpers so their function bodies see the macros (the
// preprocessor is textual).
#include "../../includes/rt/RtProfile.glsl"
#include "../../includes/rt/RtSceneSample.glsl"
#include "../../includes/rt/RtReflection.glsl"
#endif

// Screen-space reflection refinement (set 0, bindings 19/20): the *previous*
// frame's solid HDR color/depth. The proxy ray query gives plausible but
// blocky reflections; where the reflected point is on screen, marching the
// real depth buffer and sampling the real color resolves a precise mirror.
layout(set = 0, binding = 19) uniform sampler2D ssrColorTex;
layout(set = 0, binding = 20) uniform sampler2D ssrDepthTex;


// Global toggles
bool roughnessEnabled = (ubo.roughnessEnabled != 0u);
bool aoEnabled = (ubo.ambientOcclusionEnabled != 0u);

// Water fragment stage (varyings + set-2 scene textures + shadeWaterSurface
// + main). See includes/water/WaterFragStage.glsl, which includes
// includes/water/WaterSurface.glsl.
#include "../../includes/noise/Perlin.glsl"
#include "../../includes/water/WaterNoise.glsl"
#include "../../includes/water/WaterFragStage.glsl"

void main() {
    // Water surface shading writes outColor; alpha drives the blend in the
    // water pipeline (alpha-blended into the main color target).
    shadeWaterSurface();
}

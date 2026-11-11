// Hybrid RT scene bindings and lookups shared by every raster shader that
// traces inline ray queries against the real scene geometry (SolidRenderer.frag
// and the octree raymarch fragment shader).
//
// Requires the includer to:
//   * define RT_ENABLED and enable GL_EXT_ray_query at the top of the shader
//     (a mid-file #extension is illegal in GLSL);
//   * include SceneBindings.glsl / Textures.glsl / Common.glsl / Shadows.glsl
//     and RtParams.glsl before this file (the reflection helpers use ubo,
//     materials, sky, waterParams and ShadowCalculation).
//
// Declares set 0 bindings 14 (TLAS), 17 (RT params), 18 (proxy metadata),
// 21..25 (scene geometry lookups) and, for RT_PROFILE variants, 26 (counters),
// then includes the shared reflection helpers.

#ifndef RT_SCENE_BINDINGS_GLSL
#define RT_SCENE_BINDINGS_GLSL

layout(set = 0, binding = 14) uniform accelerationStructureEXT rtTlas;
layout(std140, set = 0, binding = 17) uniform RTBlock { RayTracingParams rt; };
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
// Per-op RT profiling (RT_PROFILE variants only): declares set 0 binding 26
// and the RT_PROF_* macros used by the instrumented ray-query sites. Included
// BEFORE the reflection helpers so their function bodies see the macros (the
// preprocessor is textual).
#include "RtProfile.glsl"
#include "RtSceneSample.glsl"
#include "RtReflection.glsl"

#endif // RT_SCENE_BINDINGS_GLSL

#version 460

// RT water-region depth (optional): inline ray query against the solid
// scene instance in the TES. Must precede every non-preprocessor token.
// (GL_EXT_ray_query requires GLSL 460 in glslang.)
#if defined(RT_ENABLED)
#extension GL_EXT_ray_query : require
#endif
// Shader clock for per-op RT profiling (RT_PROFILE variants only). Must be a
// top-level #extension: directives inside includes are illegal after tokens.
#ifdef RT_PROFILE
#extension GL_EXT_shader_realtime_clock : require
#endif

layout(triangles, equal_spacing, cw) in;

// ── Water tessellation evaluation shader ──────────────────────────────────
// Split out of the old shared main.tese. The stage body lives in
// includes/water/WaterTese.glsl.
#include "../../includes/SceneBindings.glsl"
#include "../../includes/Locations.glsl"

#include "../../includes/noise/Perlin.glsl"
#include "../../includes/water/WaterNoise.glsl"
#include "../../includes/water/WaterTese.glsl"

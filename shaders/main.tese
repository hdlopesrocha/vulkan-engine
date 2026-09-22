#version 460

// RT water-region depth (optional): inline ray query against the solid
// scene instance in the TES. Must precede every non-preprocessor token.
// (GL_EXT_ray_query requires GLSL 460 in glslang.)
#if defined(WATER_MODE) && defined(RT_ENABLED)
#extension GL_EXT_ray_query : require
#endif
// Shader clock for per-op RT profiling (RT_PROFILE variants only). Must be a
// top-level #extension: directives inside includes are illegal after tokens.
#ifdef RT_PROFILE
#extension GL_EXT_shader_realtime_clock : require
#endif

layout(triangles, equal_spacing, cw) in;

// Stage dispatcher (Phase-1 merge): solid terrain TES and water TES live in
// includes/solid_tese.glsl / includes/water_tese.glsl. Selected at compile
// time: water = -DWATER_MODE=1 (shaders/main_water.tese.spv).
#ifndef WATER_MODE
#define WATER_MODE 0
#endif

#include "includes/ubo.glsl"
#include "includes/locations.glsl"

#if WATER_MODE
#include "includes/perlin.glsl"
#include "includes/water_noise.glsl"
#include "includes/water_tese.glsl"
#else
#include "includes/textures.glsl"
#include "includes/common.glsl"
#include "includes/displacement.glsl"
#include "includes/solid_tese.glsl"
#endif

#version 450

layout(vertices = 3) out;

// Stage dispatcher (Phase-1 merge): the solid terrain TCS and the water TCS
// live in includes/solid_tesc.glsl / includes/water_tesc.glsl and are picked
// at COMPILE time. The water pipeline builds this source with
// -DWATER_MODE=1 (shaders/main_water.tesc.spv); terrain builds the default 0.
#ifndef WATER_MODE
#define WATER_MODE 0
#endif

#include "includes/ubo.glsl"
#include "includes/locations.glsl"

#if WATER_MODE
#include "includes/perlin.glsl"
#include "includes/water_noise.glsl"
#include "includes/water_tesc.glsl"
#else
#include "includes/solid_tesc.glsl"
#endif

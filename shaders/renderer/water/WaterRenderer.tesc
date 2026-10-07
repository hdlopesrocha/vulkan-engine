#version 450

layout(vertices = 3) out;

// ── Water tessellation control shader ─────────────────────────────────────
// Split out of the old shared main.tesc. The stage body lives in
// includes/water/WaterTesc.glsl.
#include "../../includes/SceneBindings.glsl"
#include "../../includes/Locations.glsl"

#include "../../includes/noise/Perlin.glsl"
#include "../../includes/water/WaterNoise.glsl"
#include "../../includes/water/WaterTesc.glsl"

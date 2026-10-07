#version 450

layout(vertices = 3) out;

// ── Solid terrain tessellation control shader ─────────────────────────────
// Split out of the old shared main.tesc. The stage body lives in
// includes/solid/SolidTesc.glsl.
#include "../../includes/SceneBindings.glsl"
#include "../../includes/Locations.glsl"

#include "../../includes/solid/SolidTesc.glsl"

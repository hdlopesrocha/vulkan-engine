#version 460

layout(triangles, equal_spacing, cw) in;

// ── Solid terrain tessellation evaluation shader ──────────────────────────
// Split out of the old shared main.tese. The stage body lives in
// includes/solid/SolidTese.glsl.
// The SHADOW_PASS variant was retired with the CSM shadow maps; the stage now
// feeds the regular solid/depth pipelines.
#include "../../includes/SceneBindings.glsl"
#include "../../includes/Locations.glsl"

#include "../../includes/Textures.glsl"
#include "../../includes/Common.glsl"
#include "../../includes/solid/Displacement.glsl"
#include "../../includes/solid/SolidTese.glsl"

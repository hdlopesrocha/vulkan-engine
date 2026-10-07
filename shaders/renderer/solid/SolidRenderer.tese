#version 460

layout(triangles, equal_spacing, cw) in;

// ── Solid terrain tessellation evaluation shader ──────────────────────────
// Split out of the old shared main.tese. The stage body lives in
// includes/solid/SolidTese.glsl.
// Compiled as shaders/ShadowRenderer.tese.spv with -DSHADOW_PASS=1 for the EVSM shadow
// pipeline (keeps only the displacement/position outputs).
#include "../../includes/SceneBindings.glsl"
#include "../../includes/Locations.glsl"

#include "../../includes/Textures.glsl"
#include "../../includes/Common.glsl"
#include "../../includes/solid/Displacement.glsl"
#include "../../includes/solid/SolidTese.glsl"

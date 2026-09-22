#version 450

#include "includes/locations.glsl"

// Back-face depth pass: writes water back-face depth. This pass no longer
// samples the solid scene depth — water is decoupled from the solid pass, so
// back-faces are not clipped against solids here. Water-vs-solid occlusion is
// resolved later at the composite stage (postprocess.frag), which discards
// water fragments that are behind the scene depth.

// H6 (perf report 19): VARY_UV / VARY_POSLIGHT inputs removed because the
// water TES/VS no longer produce them (they were never read here); VVL
// VUID-RuntimeSpirv-OpEntryPoint-08743 requires every declared input to have a
// matching output regardless of static use.
layout(location = VARY_LOCALPOS) in vec3 fragPos;
layout(location = VARY_NORMAL) in vec3 fragNormal;
layout(location = VARY_SHARPNORMAL) in vec3 fragBaseNormal;
layout(location = VARY_BASEPOS) in vec4 fragBasePos;
layout(location = VARY_WATERDEPTH) in float fragWaterDepth;
layout(location = VARY_SHOREDIR) in vec2 fragShoreDir;
layout(location = VARY_POSCLIP) in vec4 fragPosClip;
layout(location = VARY_DEBUG) in vec3 fragDebug;
layout(location = VARY_POSWORLD) in vec3 fragPosWorld;
layout(location = VARY_BRUSHPATCH) flat in int fragBrushIndex;
layout(location = VARY_HSV) in vec3 fragHSV;

void main() {
    // No scene-depth discard: water is independent of the solid pass. The
    // depth attachment still receives the back-face depth, which the water
    // geometry pass samples for refraction/subsurface shading.
}

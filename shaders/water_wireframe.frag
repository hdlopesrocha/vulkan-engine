#version 450

#include "includes/locations.glsl"

// Water wireframe: display bumped normal as color for visual debugging.
// H6 (perf report 19): VARY_UV / VARY_POSLIGHT inputs removed because the
// water TES no longer produces them (they were never read here); VVL
// VUID-RuntimeSpirv-OpEntryPoint-08743 requires every declared input to have a
// matching output regardless of static use. VARY_NORMAL is still read below.
layout(location = VARY_LOCALPOS) in vec3 fragPos;
layout(location = VARY_NORMAL) in vec3 fragNormal;
layout(location = VARY_SHARPNORMAL) in vec3 fragBaseNormal;
layout(location = VARY_BASEPOS) in vec4 fragBasePos;
layout(location = VARY_WATERDEPTH) in float fragWaterDepth;
layout(location = VARY_SHOREDIR) in vec2 fragShoreDir;
layout(location = VARY_POSCLIP) in vec4 fragPosClip;
layout(location = VARY_DEBUG) in vec3 fragDebug;
layout(location = VARY_POSWORLD) in vec3 fragPosWorldNotDisplaced;
layout(location = VARY_BRUSHPATCH) flat in int fragBrushIndex;
layout(location = VARY_HSV) in vec3 fragHSV;

layout(location = FRAG_OUT_COLOR) out vec4 outColor;

void main() {
    vec3 n = normalize(fragNormal);
    outColor = vec4(n * 0.5 + 0.5, 1.0);
}

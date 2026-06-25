#version 450

#include "includes/locations.glsl"

// Water wireframe: display bumped normal as color for visual debugging
layout(location = VARY_NORMAL) in vec3 fragNormal;
layout(location = VARY_POSWORLD) in vec3 fragPosWorldNotDisplaced;
layout(location = VARY_BRUSHPATCH) flat in int fragBrushIndex;

layout(location = FRAG_OUT_COLOR) out vec4 outColor;

void main() {
    vec3 n = normalize(fragNormal);
    outColor = vec4(n * 0.5 + 0.5, 1.0);
}

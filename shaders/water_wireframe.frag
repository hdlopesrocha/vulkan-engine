#version 450

#include "includes/locations.glsl"

// Minimal water-specific wireframe fragment shader: uses single brush index
layout(location = VARY_POSWORLD) in vec3 fragPosWorldNotDisplaced;
layout(location = VARY_BRUSHPATCH) flat in int fragBrushIndex;

layout(location = FRAG_OUT_COLOR) out vec4 outColor;

void main() {
    // Keep output simple for wireframe overlay
    outColor = vec4(1.0, 1.0, 1.0, 1.0);
}

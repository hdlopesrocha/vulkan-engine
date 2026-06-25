#version 450

#include "includes/locations.glsl"

// Match tessellation evaluation shader output
layout(location = VARY_NORMAL) in vec3 fragNormal; // world-space normal
layout(location = VARY_POSWORLD) in vec3 fragPosWorldNotDisplaced;
layout(location = VARY_BRUSHPATCH) flat in ivec3 fragTexIndices;

// Display normal as color for visual debugging
layout(location = FRAG_OUT_COLOR) out vec4 outColor;

void main() {
    // fragPosWorldNotDisplaced, fragTexIndices intentionally unused in wireframe pass
    vec3 n = normalize(fragNormal);
    outColor = vec4(n * 0.5 + 0.5, 1.0);
}

#version 450

#include "includes/locations.glsl"

// Match tessellation evaluation shader output
layout(location = VARY_POSWORLD) in vec3 fragPosWorldNotDisplaced;
layout(location = VARY_BRUSHPATCH) flat in ivec3 fragTexIndices;

// Minimal wireframe fragment shader: output solid white
layout(location = FRAG_OUT_COLOR) out vec4 outColor;

void main() {
    // fragPosWorldNotDisplaced intentionally unused in wireframe pass
    outColor = vec4(1.0, 1.0, 1.0, 1.0);
}

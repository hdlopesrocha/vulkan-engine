#version 450

#include "includes/locations.glsl"

// Fullscreen triangle with UV output
const vec2 POSITIONS[3] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0)
);

layout(location = VARY_UV) out vec2 uv;

void main() {
    vec2 pos = POSITIONS[gl_VertexIndex];
    gl_Position = vec4(pos, 0.0, 1.0);
    // map from clip-space [-1,1] to texture coords [0,1]
    uv = pos * 0.5 + 0.5;
}

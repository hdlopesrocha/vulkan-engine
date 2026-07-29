#version 450

// Fullscreen triangle that reconstructs world-space position from clip space.
// The sky fragment shader only needs the view direction, which we reconstruct
// using the inverse view-projection matrix already in the UBO.

#include "includes/locations.glsl"

layout(location = VARY_POSWORLD) out vec3 fragPosWorld;
layout(location = VARY_NORMAL) out vec3 fragNormal;

#include "includes/ubo.glsl"

const vec2 POSITIONS[3] = vec2[](
    vec2(-1.0, -1.0),
    vec2( 3.0, -1.0),
    vec2(-1.0,  3.0)
);

void main() {
    vec2 clipPos = POSITIONS[gl_VertexIndex];
    gl_Position = vec4(clipPos, 0.0, 1.0);

    // Reconstruct world-space position from clip space using inverse VP.
    // The w-divide gives us a point on the near plane; normalizing the
    // direction from camera to that point yields the correct view direction
    // for the sky (it is direction-only since the sky is at infinity).
    vec4 wp = ubo.invViewProjection * vec4(clipPos, 0.0, 1.0);
    fragPosWorld = wp.xyz / wp.w;
    fragNormal = vec3(0.0, 1.0, 0.0);
}

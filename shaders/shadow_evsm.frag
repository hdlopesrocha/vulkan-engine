#version 450

// Shadow VSM pass: raw depth + depth² (no exponential warp).
// Writes RGBA = (d, d², 0, 0) — cascade blending and Chebyshev
// inequality are applied during sampling.

#include "includes/locations.glsl"
#include "includes/ubo.glsl"

layout(location = VARY_POSWORLD) in vec3 inWorldPos;

layout(location = FRAG_OUT_COLOR) out vec4 outMoments;

void main() {
    vec4 lsPos = ubo.viewProjection * vec4(inWorldPos, 1.0);
    float depth = clamp(lsPos.z / lsPos.w, 0.0, 1.0);

    outMoments = vec4(depth, depth * depth, 0.0, 0.0);
}

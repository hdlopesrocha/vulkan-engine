#version 450

// Shadow EVSM2 pass: outputs EVSM moments (exp(c*d), exp(2*c*d))
// for positive-only exponential variance shadow maps with light-bleeding reduction.

#include "../../includes/Locations.glsl"
#include "../../includes/SceneBindings.glsl"
#include "../../includes/shadow/EvsmWrite.glsl"

layout(location = VARY_POSWORLD) in vec3 inWorldPos;

layout(location = FRAG_OUT_COLOR) out vec2 outEVSM;

void main() {
    vec4 lsPos = ubo.viewProjection * vec4(inWorldPos, 1.0);
    float depth = clamp(lsPos.z / lsPos.w, 0.0, 1.0);

    // Moments via the shared EVSM writer (same spelling as
    // VegetationRendererShadow.frag): 2.0 * depth is an exact power-of-two
    // scale, so exp sees bit-identical inputs and returns identical moments.
    outEVSM = evsmMoments(depth);
}

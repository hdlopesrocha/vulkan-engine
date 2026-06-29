#version 450

// Shadow EVSM pass: outputs EVSM moments (exp(c*d), exp(2*c*d), exp(-c*d), exp(-2*c*d))
// for dual-depth EVSM with light-bleeding reduction.
//
// Depth is linear [0,1] because the light projection is orthographic
// (GLM_FORCE_DEPTH_ZERO_TO_ONE ensures NDC z ∈ [0,1]).

#include "includes/locations.glsl"
#include "includes/ubo.glsl"

layout(location = VARY_POSWORLD) in vec3 inWorldPos;

layout(location = FRAG_OUT_COLOR) out vec4 outEVSM;

void main() {
    // Use viewProjection (set to current cascade's matrix per-cascade) instead
    // of lightSpaceMatrix (which always stays as cascade 0).
    vec4 lsPos = ubo.viewProjection * vec4(inWorldPos, 1.0);
    float depth = clamp(lsPos.z / lsPos.w, 0.0, 1.0);

    // EVSM warp constant.  float32 has ~7 decimal digits; with c=20 the second
    // moment at depth=1 is ~2.35e17, losing any sub-pixel variance in the lower
    // digits.  Use c=2 so moments stay small (exp(4)≈55) and variance survives.
    float c = 2.0;
    float posM1 = exp( c * depth);
    float posM2 = exp( 2.0 * c * depth);
    float negM1 = exp(-c * depth);
    float negM2 = exp(-2.0 * c * depth);

    outEVSM = vec4(posM1, posM2, negM1, negM2);
}

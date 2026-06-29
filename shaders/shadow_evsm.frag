#version 450

// Shadow EVSM pass: outputs EVSM moments (exp(c*d), exp(2*c*d), exp(-c*d), exp(-2*c*d))
// for dual-depth EVSM with light-bleeding reduction.

#include "includes/locations.glsl"
#include "includes/ubo.glsl"

layout(location = VARY_POSWORLD) in vec3 inWorldPos;

layout(location = FRAG_OUT_COLOR) out vec4 outEVSM;

void main() {
    vec4 lsPos = ubo.viewProjection * vec4(inWorldPos, 1.0);
    float depth = clamp(lsPos.z / lsPos.w, 0.0, 1.0);

    float c = 2.0;
    float posM1 = exp( c * depth);
    float posM2 = exp( 2.0 * c * depth);
    float negM1 = exp(-c * depth);
    float negM2 = exp(-2.0 * c * depth);

    outEVSM = vec4(posM1, posM2, negM1, negM2);
}

// Shared EVSM moment calculation.
// Input: depth in [0,1] range (clamped).
// Output: vec4(posM1, posM2, negM1, negM2) for dual-depth EVSM.
// Caller must declare outEVSM as layout(location = FRAG_OUT_COLOR) out vec4.

const float EVSM_WRITE_C = 2.0;

vec4 evsmMoments(float depth) {
    float posM1 = exp( EVSM_WRITE_C * depth);
    float posM2 = exp( 2.0 * EVSM_WRITE_C * depth);
    float negM1 = exp(-EVSM_WRITE_C * depth);
    float negM2 = exp(-2.0 * EVSM_WRITE_C * depth);
    return vec4(posM1, posM2, negM1, negM2);
}

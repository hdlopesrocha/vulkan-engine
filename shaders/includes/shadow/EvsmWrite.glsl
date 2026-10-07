// Shared EVSM moment calculation.
// Input: depth in [0,1] range (clamped).
// Output: vec2(posM1, posM2) for EVSM2 (positive exponential moments only).
// Caller must declare outEVSM as layout(location = FRAG_OUT_COLOR) out vec2.

const float EVSM_WRITE_C = 2.0;

vec2 evsmMoments(float depth) {
    float posM1 = exp( EVSM_WRITE_C * depth);
    float posM2 = exp( 2.0 * EVSM_WRITE_C * depth);
    return vec2(posM1, posM2);
}

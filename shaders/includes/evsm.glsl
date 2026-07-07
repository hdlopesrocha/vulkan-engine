// EVSM sampling — Chebyshev inequality on dual-depth exponential moments.
// The shadow map stores: R = exp(c*z), G = exp(2*c*z), B = exp(-c*z), A = exp(-2*c*z)
// Bias is subtracted from the test depth (making it shallower) so that
// fragments at the exact occluder depth satisfy t <= M1 → lit.

const float EVSM_C = 2.0;

float chebyshevUpperBound(vec2 moments, float t) {
    if (t <= moments.x) return 1.0;
    float variance = moments.y - moments.x * moments.x;
    variance = max(variance, 0.000001);
    float d = t - moments.x;
    float p_max = variance / (variance + d * d);
    return p_max;
}

float ShadowEVSM(sampler2D smap, vec3 projCoords, float bias) {
    float fragDepth = clamp(projCoords.z, 0.0, 1.0) - bias;

    float posDepth = exp( EVSM_C * fragDepth);
    float negDepth = exp(-EVSM_C * fragDepth);

    vec4 moments = texture(smap, projCoords.xy);

    float posProb = chebyshevUpperBound(moments.xy, posDepth);
    float negProb = chebyshevUpperBound(moments.zw, negDepth);

    float shadow = 1.0 - min(posProb, negProb);

    // Light Bleeding Reduction: eliminate the soft transition zone (halo)
    // around shadow boundaries. Values below lbr are pushed to 0 (fully lit).
    const float lbr = 0.3;
    shadow = clamp((shadow - lbr) / (1.0 - lbr), 0.0, 1.0);

    return shadow;
}

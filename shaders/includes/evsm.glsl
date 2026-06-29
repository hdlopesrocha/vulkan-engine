// EVSM sampling — Chebyshev inequality on dual-depth moments.
// The shadow map stores: R = exp(c*z), G = exp(2*c*z), B = exp(-c*z), A = exp(-2*c*z)

const float EVSM_C = 2.0;

// Positive warp Chebyshev upper bound: probability fragment is LIT.
// moments.xy = (M1, M2) for the positive warp, t = exp(c * fragDepth)
float chebyshevUpperBound(vec2 moments, float t) {
    if (t <= moments.x) return 1.0;
    float variance = moments.y - moments.x * moments.x;
    variance = max(variance, 0.00001);
    float d = t - moments.x;
    float p_max = variance / (variance + d * d);
    return p_max;
}

// Sample EVSM texture with bilinear filtering and return shadow factor [0, 1]
// (0 = fully lit, 1 = fully shadowed)
//
// Bias is applied by SHRINKING the test depth (dividing by exp(c*bias)) so
// that fragments at the exact surface depth satisfy t <= moments.x → lit.
// This matches PCF's bias semantics (bias pushes the stored depth forward,
// making it easier for the fragment to be "in front").
float ShadowEVSM(sampler2D smap, vec3 projCoords, float bias) {
    float fragDepth = clamp(projCoords.z, 0.0, 1.0);
    vec4 moments = texture(smap, projCoords.xy);

    float biasMul = exp(EVSM_C * bias);
    float posDepth = exp(EVSM_C * fragDepth) / biasMul;
    float negDepth = exp(-EVSM_C * fragDepth) / biasMul;

    float posProb = chebyshevUpperBound(moments.xy, posDepth);
    float negProb = chebyshevUpperBound(moments.zw, negDepth);

    float shadow = min(posProb, negProb);
    return 1.0 - shadow;
}

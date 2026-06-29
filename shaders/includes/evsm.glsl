// VSM sampling (no exponential warp) — Chebyshev inequality on raw depth moments.
// The shadow map stores: R = depth, G = depth², B/A unused.
//
// Bias is SUBTRACTED from the test depth (making it shallower) so that
// fragments at the exact occluder depth satisfy t <= M1 → lit.  This
// mirrors PCF's bias semantics.

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
    vec4 moments = texture(smap, projCoords.xy);

    float prob = chebyshevUpperBound(moments.xy, fragDepth);
    return 1.0 - prob;
}

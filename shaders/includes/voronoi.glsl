// Voronoi (Worley) noise helpers — returns nearest (F1) and second-nearest (F2) distances.
// Returns: vec2(F1, F2)
vec3 hash3(vec3 p) {
    return fract(sin(vec3(
        dot(p, vec3(127.1, 311.7, 74.7)),
        dot(p, vec3(269.5, 183.3, 246.1)),
        dot(p, vec3(113.5, 271.9, 124.6))
    )) * 43758.5453123);
}

// voronoi3d(x, time, fbmSpatialScale, fbmTimeScale, fbmOctaves, fbmPersistence)
// `time` and FBM parameters are used to compute a small jitter for each
// feature point using the same FBM used elsewhere (via `waterFbmNoise`). This
// perturbs the feature positions (point jitter) rather than translating the
// entire Voronoi lattice, producing animated but locally coherent motion.
vec2 voronoi3d(vec3 x, float time, float fbmSpatialScale, float fbmTimeScale, int fbmOctaves, float fbmPersistence) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    float min1 = 1e10;
    float min2 = 1e10;
    // Jitter amplitude in cell-space (0..1). Tunable constant; keeps points
    // within or near their cell while allowing noticeable motion.
    const float jitterAmp = 0.4;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            for (int k = -1; k <= 1; ++k) {
                vec3 b = vec3(float(i), float(j), float(k));
                // Base feature point inside cell
                vec3 baseRp = hash3(p + b);

                // Sample FBM once per feature to get a scalar modulation in [-1..1]
                // Sample position uses the integer cell coord plus the base feature
                // offset so neighbouring cells produce different samples.
                float fbmSample = waterFbmNoise(p + b + baseRp, fbmSpatialScale, time, fbmTimeScale, fbmOctaves, fbmPersistence, vec3(0.0));

                // Direction for jitter derived from a separate hash so it's stable
                // but decorrelated from baseRp. Map to [-1,1] then normalize.
                vec3 dir = normalize(hash3(p + b + vec3(17.0)) * 2.0 - 1.0);

                // Apply jitter scaled by FBM sample and amplitude
                vec3 rp = baseRp + dir * (fbmSample * jitterAmp);

                vec3 pos = b + rp - f;
                float d = dot(pos, pos);
                if (d < min1) {
                    min2 = min1;
                    min1 = d;
                } else if (d < min2) {
                    min2 = d;
                }
            }
        }
    }
    return vec2(sqrt(min1), sqrt(min2));
}

// Best-view select shared by the impostor color + depth vertex twins.
// Extracted verbatim from VegetationRendererImpostor.vert (perf report 26 H4):
// the winning Fibonacci view depends only on instance rotation + camera
// direction, so every corner of a quad computes the same layer. The loop
// still runs once per vertex (a true once-per-quad broadcast would need
// subgroup ops, and quad-verts-in-one-subgroup is not portable), but both
// twins now share one spelling.
// BIT-IDENTICAL CONTRACT: same (toCamera, rotFrac, billboardIdx) in gives
// the same layer out — the same float ops in the same order as the inline
// loop this replaces. Pure function of its arguments; requires nothing.
int impostorBestLayerIndex(vec3 toCamera, float rotFrac, int billboardIdx) {
    const float goldenAngle = 3.14159265358979323846 * (3.0 - 2.2360679774997896);
    const int NUM_VIEWS = 20;

    float instTheta = rotFrac * 6.28318530718;
    float cI = cos(instTheta);
    float sI = sin(instTheta);
    vec3 toCamera_canonical = vec3(
         cI * toCamera.x + sI * toCamera.z,
        toCamera.y,
        -sI * toCamera.x + cI * toCamera.z
    );

    int bestIdx = 0;
    float bestDot = -2.0;
    for (int i = 0; i < NUM_VIEWS; i++) {
        float y = 1.0 - (float(i) + 0.5) / float(NUM_VIEWS) * 2.0;
        float r = sqrt(max(0.0, 1.0 - y * y));
        float theta = goldenAngle * float(i);
        vec3 d = vec3(cos(theta) * r, y, sin(theta) * r);
        float dt = dot(toCamera_canonical, d);
        if (dt > bestDot) { bestDot = dt; bestIdx = i; }
    }

    return clamp(billboardIdx, 0, 2) * NUM_VIEWS + bestIdx;
}

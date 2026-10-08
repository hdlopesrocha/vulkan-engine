#ifndef SDF_GRASS_GLSL
#define SDF_GRASS_GLSL

// Procedural grouped grass clump (SDF_PRIM_GRASS).
//
// ONE SdfInstance per existing vegetation instance: the shader expands it
// into a small, deterministic group of curved blades instead of treating
// every blade as its own SDF object. The returned distance is a true (or
// conservative) distance for the union of the evaluated primitives, so the
// generic raymarcher's sphere tracing strides over empty space aggressively:
//
//   1. an aggregate envelope (one tapered round cone) contains the whole
//      clump — samples farther than the detail range get this distance and
//      skip the entire clump without evaluating a blade;
//   2. near the envelope the blades are evaluated (two round cones each,
//      both exact), and the blade count falls with the clump's projected-size
//      proxy (camera distance in clump scales);
//   3. beyond GRASS_COARSE_SCALE the clump stays on the aggregate envelope,
//      which is the far-LOD SDF representation (no impostor/billboard swap
//      inside the SDF path).
//
// Wind is applied as a rigid rotation of the clump axis (plus a per-blade
// bend modulation); rotations preserve distances, so no Lipschitz
// compensation is needed for the lean and sphere tracing never oversteps.
//
// params0 = (clumpRadius, bladeHeight, bladeWidth, bladeCount)
// params1 = (curvature, maxLean, windGain, tipWidth)
// Instance seed + index derive every per-blade variation (no stored data).

// Cheap deterministic hash (integer mix of the float bits). Avoids sin/cos
// and per-frame RNG; the same instance always yields identical blades.
float sdfGrassHash1(float n) {
    uint u = floatBitsToUint(n + 0.1234567);
    u = (u ^ (u >> 16u)) * 0x7feb352du;
    u = (u ^ (u >> 15u)) * 0x846ca68bu;
    u = u ^ (u >> 16u);
    return float(u) * (1.0 / 4294967295.0);
}

vec2 sdfGrassHash2(float n) {
    return vec2(sdfGrassHash1(n), sdfGrassHash1(n + 17.357));
}

// Exact round cone (IQ): cone from a (radius r1) to b (radius r2). Falls back
// to the containing capsule when the taper is degenerate (a2 <= 0): a
// superset of the round cone, so its (smaller) distance stays conservative
// for sphere tracing.
float sdGrassRoundCone(vec3 p, vec3 a, vec3 b, float r1, float r2) {
    vec3 ba = b - a;
    float l2 = dot(ba, ba);
    float rr = r1 - r2;
    float a2 = l2 - rr * rr;
    if (l2 < 1e-10 || a2 < 1e-8) {
        vec3 pa = p - a;
        float h = (l2 > 1e-10) ? clamp(dot(pa, ba) / l2, 0.0, 1.0) : 0.0;
        return length(pa - ba * h) - max(r1, r2);
    }
    float il2 = 1.0 / l2;
    vec3 pa = p - a;
    float y = dot(pa, ba);
    float z = y - l2;
    vec3 q = pa * l2 - ba * y;
    float x2 = dot(q, q);
    float y2 = y * y * l2;
    float z2 = z * z * l2;
    float k = sign(rr) * rr * rr * x2;
    if (sign(z) * a2 * z2 > k) return sqrt(x2 + z2) * il2 - r2;
    if (sign(y) * a2 * y2 < k) return sqrt(x2 + y2) * il2 - r1;
    return (sqrt(x2 * a2 * il2) + y * rr) * il2 - r1;
}

// One curved blade: two chained round cones (root -> mid -> tip) tapering
// from `width` to width * tipFrac, bent sideways by `curve` x height along
// the unit horizontal direction `bendDir`. Both segments are exact, so the
// min is an exact distance for the union.
float sdGrassBlade(vec3 p, vec3 root, vec3 up, float h, float width,
                   float tipFrac, float curve, vec2 bendDir) {
    float hh = max(h, 1e-4);
    float sideways = curve * hh;
    vec3 bd = vec3(bendDir.x, 0.0, bendDir.y);
    vec3 mid = root + up * (hh * 0.5) + bd * (sideways * 0.35);
    vec3 tip = root + up * hh + bd * sideways;
    float r0 = max(width, 1e-5);
    float r1 = mix(r0, r0 * tipFrac, 0.5);
    float r2 = max(r0 * tipFrac, 1e-5);
    return min(sdGrassRoundCone(p, root, mid, r0, r1),
               sdGrassRoundCone(p, mid, tip, r1, r2));
}

// Grouped clump SDF. `camScale` = camera distance in clump scales (the
// projected-size proxy), `windDir` = unit local wind direction in XZ,
// `windAmp` = clamped lean angle (radians).
float sdGrassClump(vec3 p, vec4 p0, vec4 p1, float seed, float camScale,
                   vec2 windDir, float windAmp) {
    // Constants: detail switch radius, blade-count LOD scales and the
    // aggregate-only camera scale.
    const float GRASS_DETAIL_MIN = 0.35;  // x height, floor for the detail range
    const float GRASS_DETAIL_WIDTH = 6.0; // x width, floor for the detail range
    const float GRASS_LOD_NEAR = 16.0;    // camera scale where blade count drops to 8
    const float GRASS_LOD_FAR = 32.0;     // camera scale where blade count drops to 4
    const float GRASS_COARSE_SCALE = 60.0; // camera scale for aggregate-only LOD
    // Per-blade curvature multiplier ceiling: blades use curve * [0.6, 1.4],
    // so the aggregate envelope (and the CPU AABB in SdfScene.cpp) must cover
    // 1.4 x curvature or the early-out could overestimate near the tips.
    const float GRASS_BEND_MAX = 1.4;

    float radius = max(p0.x, 1e-3);
    float height = max(p0.y, 1e-3);
    float width = clamp(p0.z, 1e-5, max(radius * 0.5, 1e-4));
    int count = int(clamp(p0.w, 1.0, 64.0));
    float curve = clamp(p1.x, 0.0, 1.5);
    float tipFrac = clamp(p1.w, 0.05, 1.0);

    // Rigid wind lean of the clump axis toward the local wind direction.
    vec3 up = vec3(0.0, 1.0, 0.0);
    float lean = clamp(windAmp, 0.0, 1.2);
    if (lean > 1e-4) {
        vec2 wd = windDir;
        float wl2 = dot(wd, wd);
        if (wl2 > 1e-8) {
            wd *= inversesqrt(wl2);
            // Rodrigues rotation of +Y about cross(+Y, wd) (axis is
            // perpendicular to Y, so k·Y = 0).
            vec3 axis = vec3(wd.y, 0.0, -wd.x);
            float c = cos(lean);
            float s = sin(lean);
            up = normalize(vec3(0.0, c, 0.0) + cross(axis, vec3(0.0, 1.0, 0.0)) * s);
        }
    }

    // Aggregate envelope: one tapered round cone that contains every blade
    // (roots within `radius`, tips bent by at most 1.4 x height * curve
    // relative to the leaned axis). Exact distance for the envelope shape, so
    // returning it to the raymarcher is safe at any distance.
    float envR0 = radius + 2.0 * width;
    float envR1 = radius + 2.0 * width + height * curve * GRASS_BEND_MAX;
    float agg = sdGrassRoundCone(p, vec3(0.0), up * height, envR0, envR1);

    // Empty-space skipping: beyond the detail range the envelope distance is
    // the result, so the march crosses the whole clump in one step.
    float detail = max(height * GRASS_DETAIL_MIN, width * GRASS_DETAIL_WIDTH);
    if (agg > detail) return agg;

    // Aggregate-only far LOD (still an SDF, no impostor/billboard).
    if (camScale > GRASS_COARSE_SCALE) return agg;

    // Blade-count LOD: deterministic hash-ordered subset, stable per frame.
    int n = count;
    if (camScale > GRASS_LOD_FAR) n = min(n, 4);
    else if (camScale > GRASS_LOD_NEAR) n = min(n, 8);

    float d = 1e5;
    for (int i = 0; i < n; ++i) {
        float fi = float(i) + seed * 37.0;
        // Root scatter: deterministic azimuth + radial distance.
        vec2 rh = sdfGrassHash2(fi);
        vec2 dir2 = vec2(rh.x * 2.0 - 1.0, rh.y * 2.0 - 1.0);
        float dl = length(dir2);
        dir2 = (dl > 1e-4) ? dir2 / dl : vec2(1.0, 0.0);
        vec3 root = vec3(dir2.x, 0.0, dir2.y) * (radius * sdfGrassHash1(fi + 5.0));
        // Per-blade bend direction (unit horizontal).
        vec2 bh = sdfGrassHash2(fi + 11.0);
        vec2 bend = vec2(bh.x * 2.0 - 1.0, bh.y * 2.0 - 1.0);
        float bl = length(bend);
        bend = (bl > 1e-4) ? bend / bl : vec2(1.0, 0.0);
        float bhgt = sdfGrassHash1(fi + 3.0);
        float bwr = sdfGrassHash1(fi + 7.0);
        d = min(d, sdGrassBlade(p, root, up,
                                height * (0.55 + 0.45 * bhgt),
                                width * (0.6 + 0.8 * bwr),
                                tipFrac,
                                curve * (0.6 + 0.8 * bh.x),
                                bend));
    }
    return d;
}

#endif // SDF_GRASS_GLSL

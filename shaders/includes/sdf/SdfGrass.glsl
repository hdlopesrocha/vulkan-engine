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
//   2. near the envelope the blades are evaluated (two cheap conservative
//      tapered segments each), and the blade count falls with the clump's
//      projected-size proxy (camera distance in clump scales);
//   3. beyond GRASS_IMPOSTOR_START a dedicated aggregate impostor
//      (sdGrassImpostor: one base-bulged round cone + a one-signed
//      low-frequency ripple) fades in as a conservative offset of the union:
//          d = min(bladeD, impostorD + (1 - t) * K)
//      with t the camera-scale fraction across [START, FULL]. K exceeds the
//      impostor inradius, so at t = 0 the offset surface is empty and the
//      union is exactly the blade set (no pop entering the band); at t = 1
//      the offset is zero and the impostor (a superset of the blade
//      envelope) is fully present. d + offset is a valid SDF of the inward
//      offset surface of a distance field, so every step stays conservative.
//      At FULL and beyond the impostor is the whole field and no blade is
//      ever evaluated (hard cheap far path). A shadow march with a fixed
//      camScale >= FULL therefore evaluates the impostor only.
//
// Wind is a rigid rotation of the clump axis (rotations preserve distances)
// plus a per-blade bend direction blended toward the local wind vector; the
// blend of two unit vectors stays inside the unit disk, so the blade tips
// stay within the same curvature budget the envelope and the CPU AABB cover.
// The impostor uses the same leaned axis, so its mass follows the wind.
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
// for sphere tracing. Only the aggregate envelope uses it: the exact
// distance keeps the early-out and the far-LOD silhouette tight.
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

// Cheap conservative tapered segment: the blade's building block. `n` is the
// unit axis a->b, `invL` = 1/|b-a|, `r0 + dr*h` the interpolated radius at
// the clamped projection, `cosA` the cosine of the cone half-angle
// (sqrt(1 - (dr/L)^2)).
//
// The exact round-cone distance is min over t of |p - c(t)| - r(t). Instead
// of that three-way piecewise exact evaluation this measures the distance to
// the TANGENT cone through the clamped projection: length(pa - n*dot(pa,n))
// is the radial distance to the axis, scaled by cosA it is the distance to
// the tangent lateral surface, and subtracting the interpolated radius at
// the clamped projection keeps it exact on that surface. The tangent plane
// is a supporting plane of the convex hull of the two end spheres, so the
// result is always <= the exact distance and sphere tracing cannot overstep.
// Near the end caps it under-estimates by at most r*(1-cosA)/cosA (the zero
// set is the same cone with cap radii inflated by 1/cosA, which the width
// clamp in sdGrassClump keeps inside the envelope's 2x width margin).
float sdGrassTaper(vec3 p, vec3 a, vec3 n, float invL, float r0, float dr, float cosA) {
    vec3 pa = p - a;
    float t = dot(pa, n);
    float h = clamp(t * invL, 0.0, 1.0);
    return length(pa - n * t) * cosA - r0 - dr * h;
}

// One curved blade: two chained cheap tapered segments (root -> mid -> tip)
// tapering from `width` to width * tipFrac, bent sideways by `curve` x height
// along the horizontal direction `bendDir` (|bendDir| <= 1, so the tip stays
// within curve * height of the axis). Each segment is a conservative bound,
// so the min is a conservative distance for the union.
float sdGrassBlade(vec3 p, vec3 root, vec3 up, float h, float width,
                   float tipFrac, float curve, vec2 bendDir) {
    float hh = max(h, 1e-4);
    vec3 bd = vec3(bendDir.x, 0.0, bendDir.y);
    vec3 mid = root + up * (hh * 0.5) + bd * (curve * hh * 0.35);
    vec3 tip = root + up * hh + bd * (curve * hh);
    float r0 = max(width, 1e-5);
    float r1 = mix(r0, r0 * tipFrac, 0.5);
    float r2 = max(r0 * tipFrac, 1e-5);

    // Per-segment slope factor sqrt(1 - slope^2), slope = dr/L.
    vec3 ba1 = mid - root;
    vec3 ba2 = tip - mid;
    float invL1 = inversesqrt(max(dot(ba1, ba1), 1e-12));
    float invL2 = inversesqrt(max(dot(ba2, ba2), 1e-12));
    float s1 = (r1 - r0) * invL1;
    float s2 = (r2 - r1) * invL2;
    float c1 = sqrt(max(1.0 - s1 * s1, 0.0));
    float c2 = sqrt(max(1.0 - s2 * s2, 0.0));
    float d1 = sdGrassTaper(p, root, ba1 * invL1, invL1, r0, r1 - r0, c1);
    float d2 = sdGrassTaper(p, mid, ba2 * invL2, invL2, r1, r2 - r1, c2);
    return min(d1, d2);
}

// Lipschitz constant of the rippled cone (coneD - disp): 1 + amp * (freq +
// 2/height). Dividing by it makes sdGrassImpostor 1-Lipschitz; sdGrassClump
// also uses it to convert the impostor inradius bound into a fade offset.
float sdGrassImpostorLip(float height, float amp, float freq) {
    return 1.0 + amp * (freq + 2.0 / max(height, 1e-3));
}

// Aggregate impostor (far-LOD clump mass): the caller's exact round-cone
// distance `d` displaced outward by a low-frequency, one-signed ripple. One
// primitive + two sines; no blade is ever evaluated on this path.
//
// Conservative for sphere tracing: the ripple is bounded (0 <= disp <= amp)
// and its gradient is bounded (|grad disp| <= amp * (freq + 2/height), with
// the clamp on the height fraction not adding slope), so (d - disp) has
// Lipschitz constant <= 1 + amp * (freq + 2/height). Dividing by that
// constant yields a 1-Lipschitz field: |f(p)| <= dist(p, zero set), so the
// value can never overestimate the distance to its own zero set and the
// sphere tracer cannot overstep. Because disp is one-signed outward, the
// impostor volume contains the cone volume (and with it the blade envelope
// and every blade surface), so the blade term can be dropped at the far
// LOD without popping a blade.
//
// `amp` = maximum outward ripple, `freq` = ripple cycles per local unit.
// The ripple is strongest at the base and fades quadratically to zero at
// the top (base concentration, upper taper); the seed phases it per clump.
float sdGrassImpostor(vec3 p, vec3 up, float height, float d, float seed,
                      float amp, float freq) {
    float hh = max(height, 1e-3);
    // Height fraction along the (leaned) clump axis.
    float yw = clamp(dot(p, up) / hh, 0.0, 1.0);
    // Per-clump phase: the same instance seed always yields the same mass.
    float ph = seed * 12.9898;
    float s = sin(p.x * freq + ph) * sin(p.z * freq + ph * 1.37 + 1.7);
    float disp = amp * (0.5 + 0.5 * s) * (1.0 - yw * yw);
    return (d - disp) / sdGrassImpostorLip(height, amp, freq);
}

// Grouped clump SDF. `camScale` = camera distance in clump scales (the
// projected-size proxy), `windDir` = unit local wind direction in XZ,
// `windAmp` = clamped lean angle (radians).
float sdGrassClump(vec3 p, vec4 p0, vec4 p1, float seed, float camScale,
                   vec2 windDir, float windAmp) {
    // Constants: detail switch radius, blade-count LOD scales, the impostor
    // fade band and the impostor shape. The impostor shape constants have CPU
    // twins in SdfScene.cpp (localHalfExtents) that pad the instance AABB and
    // must stay in sync.
    const float GRASS_DETAIL_MIN = 0.35;  // x height, floor for the detail range
    const float GRASS_DETAIL_WIDTH = 6.0; // x width, floor for the detail range
    const float GRASS_LOD_NEAR = 16.0;    // camera scale where blade count drops to 8
    const float GRASS_LOD_FAR = 32.0;     // camera scale where blade count drops to 4
    // Impostor fade band: at START the offset keeps the impostor fully hidden
    // (offset > its inradius), at FULL it is the whole field (zero blades).
    // A shadow march with a fixed camScale >= FULL evaluates the impostor
    // only, the cheapest shadow representation.
    const float GRASS_IMPOSTOR_START = 48.0;
    const float GRASS_IMPOSTOR_FULL = 80.0;
    // Impostor shape: base bulge as a fraction of the envelope's base->top
    // radius growth (base concentration), ripple amplitude as a fraction of
    // max(radius, height/2) and ripple cycles per clump scale (low
    // frequency). Mirrored in SdfScene.cpp for the CPU AABB.
    const float GRASS_IMPOSTOR_BASE_BULGE = 0.5;
    const float GRASS_IMPOSTOR_RIPPLE = 0.30;
    const float GRASS_IMPOSTOR_RIPPLE_FREQ = 2.5;
    // Per-blade curvature multiplier ceiling: blades use curve * [0.6, 1.4],
    // so the aggregate envelope (and the CPU AABB in SdfScene.cpp) must cover
    // 1.4 x curvature or the early-out could overestimate near the tips.
    const float GRASS_BEND_MAX = 1.4;

    float radius = max(p0.x, 1e-3);
    float height = max(p0.y, 1e-3);
    // Width is capped against the clump radius and the blade height: the
    // height cap keeps the tapered segments from degenerating (slope < 1) so
    // the conservative cosA factor stays well defined and the bound's cap
    // inflation (1/cosA <= ~1.3) stays inside the envelope's 2x width margin.
    float width = clamp(p0.z, 1e-5,
                        min(max(radius * 0.5, 1e-4), max(height * 0.25, 1e-4)));
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
    float detail = max(height * GRASS_DETAIL_MIN, width * GRASS_DETAIL_WIDTH);

    // Impostor shape (shared by the fade band and the hard far path): a
    // base-bulged envelope cone, so the base is concentrated while the top
    // keeps the envelope's upper taper.
    float impScale = max(radius, height * 0.5);
    float impAmp = GRASS_IMPOSTOR_RIPPLE * impScale;
    float impFreq = GRASS_IMPOSTOR_RIPPLE_FREQ / max(impScale, 1e-3);
    float impR0 = envR0 + GRASS_IMPOSTOR_BASE_BULGE * (envR1 - envR0);
    float impR1 = envR1;

    // Hard far LOD (at FULL and beyond): impostor only, no blade is ever
    // evaluated. The impostor contains the blade envelope (see
    // sdGrassImpostor), so dropping the blade term cannot pop a blade
    // surface.
    if (camScale >= GRASS_IMPOSTOR_FULL) {
        float icd = sdGrassRoundCone(p, vec3(0.0), up * height, impR0, impR1);
        // Empty-space skip: the one-signed ripple extends the mass outward by
        // at most impAmp, so icd - impAmp stays below the true distance to
        // the impostor zero set. (No Lipschitz division needed here: the skip
        // value only has to bound the true distance.)
        if (icd > max(detail, impAmp * 1.5)) return icd - impAmp;
        return sdGrassImpostor(p, up, height, icd, seed, impAmp, impFreq);
    }

    // Empty-space skipping: beyond the detail range the envelope distance is
    // the result, so the march crosses the whole clump in one step. Once the
    // impostor is active the skip must bound BOTH union terms (the base bulge
    // reaches past the envelope), so it returns the smaller of the envelope
    // distance and the ripple-bounded impostor distance; a non-positive
    // bound falls through to the full evaluation instead of reporting a
    // false interior.
    float agg = sdGrassRoundCone(p, vec3(0.0), up * height, envR0, envR1);
    bool impostorActive = (camScale > GRASS_IMPOSTOR_START);
    float icd = 0.0;
    float impOff = 0.0;
    if (impostorActive) {
        // Fade the impostor in with a conservative offset. The compensated
        // impostor field (d - disp) / lip has inradius <= (max(impR0, impR1)
        // + impAmp) / lip, so K is that bound: at START impostorD + K is
        // strictly positive everywhere (empty offset surface, the union is
        // exactly the blade set) and the offset surface emerges just inside
        // the band, spreading the whole morph over [START, FULL] instead of
        // popping it out near the end. At FULL the offset is zero and the
        // impostor is fully present. d + off is a valid SDF of the inward
        // offset surface of d, so the union stays conservative at every t.
        float t = clamp((camScale - GRASS_IMPOSTOR_START) /
                        (GRASS_IMPOSTOR_FULL - GRASS_IMPOSTOR_START), 0.0, 1.0);
        float impK = (max(impR0, impR1) + impAmp) /
                     sdGrassImpostorLip(height, impAmp, impFreq) + 1e-3;
        impOff = (1.0 - t) * impK;
        icd = sdGrassRoundCone(p, vec3(0.0), up * height, impR0, impR1);
    }
    if (agg > detail) {
        if (!impostorActive) return agg;
        float bound = min(agg, icd - impAmp + impOff);
        if (bound > 0.0) return bound;
    }

    // Blade-count LOD: deterministic hash-ordered subset, stable per frame.
    int n = count;
    if (camScale > GRASS_LOD_FAR) n = min(n, 4);
    else if (camScale > GRASS_LOD_NEAR) n = min(n, 8);

    float d = 1e5;
    for (int i = 0; i < n; ++i) {
        float fi = float(i) + seed * 37.0;
        // Root azimuth: a uniform XZ direction from one hash pair. The unit
        // direction doubles as the blade's outward bend, so no second
        // direction hash and no second normalize are needed.
        vec2 rh = sdfGrassHash2(fi) * 2.0 - 1.0;
        float rl = length(rh);
        rh = (rl > 1e-4) ? rh / rl : vec2(1.0, 0.0);
        // Remaining per-blade scalars from a second hash pair; the curvature
        // and wind phase are cheap fract remixes instead of more hashes.
        vec2 sh = sdfGrassHash2(fi + 13.7);
        vec3 root = vec3(rh.x, 0.0, rh.y) * (radius * sh.x);
        // Per-blade wind: blend the outward bend direction toward the local
        // wind direction. Both inputs are unit, so the blend stays inside the
        // unit disk and the tip offset never exceeds the GRASS_BEND_MAX
        // budget the envelope and the CPU AABB already cover.
        vec2 bend = mix(rh, windDir,
                        clamp(windAmp * (1.0 + fract(sh.x * 3.77)), 0.0, 1.0));
        d = min(d, sdGrassBlade(p, root, up,
                                height * (0.55 + 0.45 * sh.y),
                                width * (0.6 + 0.8 * fract(sh.x * 7.31)),
                                tipFrac,
                                curve * (0.6 + 0.8 * fract(sh.y * 5.17)),
                                bend));
    }
    if (impostorActive) {
        // Union of the blade set and the faded impostor zero set; both terms
        // are conservative, so the min is conservative for the union.
        d = min(d, sdGrassImpostor(p, up, height, icd, seed, impAmp, impFreq) + impOff);
    }
    return d;
}

#endif // SDF_GRASS_GLSL

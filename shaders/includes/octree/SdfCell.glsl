// Bit-exact ports of the CPU cell functions the offline Blender exporter uses:
//
//   SDF::eval()                  -> sdfEvalType()
//   SDF::getPosition()           -> sdfCellPosition()
//   SDF::getNormalFromPosition() -> sdfCellNormal()
//   Math::solveLinearSystem()    -> sdfSolveLinearSystem() (Cramer, glm::det order)
//
// This is the runtime's ONLY Cartesian cell solver (OctreeTriangles.comp calls
// it). The CPU implementations exist for the offline Blender exporter only;
// the engine never solves a chunk vertex on the host and never reads one back.
//
// The functions are pure (they take the 8 stored corner values and the cell
// cube, not a node table), so a caller with any storage layout can use them.
//
// Two details make the results bit-identical to the CPU rather than merely
// close:
//   * `precise` on the sensitive expressions, so the compiler may not
//     fuse/reassociate them (the QEF below is near-singular for cells with a
//     single negative corner, which amplifies last-bit differences);
//   * GLSL builtins are only required to be accurate to a few ulp, so
//     dot/length/outerProduct are spelled out exactly as GLM evaluates them.

#ifndef SDF_CELL_GLSL
#define SDF_CELL_GLSL

// Positive infinity as a bit pattern: comparing against it keeps the sentinel
// test exact.
#define SDF_INF (uintBitsToFloat(0x7F800000u))

const int SDF_TYPE_EMPTY = 0;
const int SDF_TYPE_SURFACE = 1;
const int SDF_TYPE_SOLID = 2;

// CUBE_CORNERS: bit2 = x, bit1 = y, bit0 = z (math/BoundingCube.hpp).
const vec3 SDF_CELL_CORNERS[8] = vec3[](
    vec3(0, 0, 0), vec3(0, 0, 1), vec3(0, 1, 0), vec3(0, 1, 1),
    vec3(1, 0, 0), vec3(1, 0, 1), vec3(1, 1, 0), vec3(1, 1, 1)
);
// SDF_EDGES (sdf/SDF.hpp) — same order as the CPU QEF loop.
const ivec2 SDF_CELL_EDGES[12] = ivec2[](
    ivec2(0, 1), ivec2(1, 3), ivec2(2, 3), ivec2(0, 2),
    ivec2(4, 5), ivec2(5, 7), ivec2(6, 7), ivec2(4, 6),
    ivec2(0, 4), ivec2(1, 5), ivec2(2, 6), ivec2(3, 7)
);

bool sdfIsFinite(float v) { return !(isnan(v) || isinf(v)); }

vec3 sdfCellCorner(vec3 mn, float sz, int idx) { return mn + SDF_CELL_CORNERS[idx] * sz; }

// SDF::eval — sign classification with the INFINITY sentinel skipped.
int sdfEvalType(float sdf[8]) {
    bool hasPositive = false;
    bool hasNegative = false;
    for (int i = 0; i < 8; ++i) {
        float v = sdf[i];
        if (v == SDF_INF) {
            continue; // skip untouched corners
        } else if (v >= 0.0f) {
            hasPositive = true;
        } else {
            hasNegative = true;
        }
    }
    if (!hasPositive && !hasNegative) return SDF_TYPE_EMPTY;
    return (hasNegative && hasPositive) ? SDF_TYPE_SURFACE : (hasPositive ? SDF_TYPE_EMPTY : SDF_TYPE_SOLID);
}

// SDF::getNormalFromPosition — analytic trilinear gradient at `position`.
vec3 sdfCellNormal(float sdf[8], vec3 mn, float sz, vec3 position) {
    precise vec3 local = (position - mn) / sz;
    float s0 = sdf[0], s1 = sdf[1], s2 = sdf[2], s3 = sdf[3];
    float s4 = sdf[4], s5 = sdf[5], s6 = sdf[6], s7 = sdf[7];
    precise float dx = (
        (1.0 - local.y) * (1.0 - local.z) * (s4 - s0) +
        (1.0 - local.y) * local.z * (s5 - s1) +
        local.y * (1.0 - local.z) * (s6 - s2) +
        local.y * local.z * (s7 - s3)
    );
    precise float dy = (
        (1.0 - local.x) * (1.0 - local.z) * (s2 - s0) +
        local.x * (1.0 - local.z) * (s3 - s1) +
        (1.0 - local.x) * local.z * (s6 - s4) +
        local.x * local.z * (s7 - s5)
    );
    precise float dz = (
        (1.0 - local.x) * (1.0 - local.y) * (s1 - s0) +
        (1.0 - local.x) * local.y * (s3 - s2) +
        local.x * (1.0 - local.y) * (s5 - s4) +
        local.x * local.y * (s7 - s6)
    );
    precise vec3 normal = vec3(dx, dy, dz) / sz;
    precise float nl = sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    if (!sdfIsFinite(nl) || nl < 1e-6) {
        return vec3(0.0, 1.0, 0.0);
    }
    return normal / nl;
}

// glm::determinant(mat3) — column-major indexing m[col][row].
float sdfDet3(mat3 m) {
    return m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2])
         - m[1][0] * (m[0][1] * m[2][2] - m[2][1] * m[0][2])
         + m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2]);
}

// Math::solveLinearSystem (Cramer's rule, first column replaced).
vec3 sdfSolveLinearSystem(mat3 A, vec3 b) {
    precise float detA = sdfDet3(A);
    if (abs(detA) < 1e-6) {
        return vec3(0.0);
    }
    mat3 A1 = A;
    mat3 A2 = A;
    mat3 A3 = A;
    A1[0] = b;
    A2[1] = b;
    A3[2] = b;
    precise float x = sdfDet3(A1) / detA;
    precise float y = sdfDet3(A2) / detA;
    precise float z = sdfDet3(A3) / detA;
    return vec3(x, y, z);
}

// SDF::getPosition — QEF with corner snap, sentinel guard and fallbacks.
vec3 sdfCellPosition(float sdf[8], vec3 mn, float sz) {
    if (sdfEvalType(sdf) != SDF_TYPE_SURFACE) {
        return mn + vec3(sz * 0.5); // cube centre
    }
    // Exact-zero corner: snap onto it so neighbouring cells sharing the corner
    // emit the identical position (the CPU does the same).
    for (int i = 0; i < 8; ++i) {
        if (sdf[i] == 0.0) {
            return sdfCellCorner(mn, sz, i);
        }
    }

    vec3 normals[8];
    for (int i = 0; i < 8; ++i) {
        normals[i] = sdfCellNormal(sdf, mn, sz, sdfCellCorner(mn, sz, i));
    }

    precise mat3 ATA = mat3(0.0);
    precise vec3 ATb = vec3(0.0);
    precise vec3 massPoint = vec3(0.0);
    int count = 0;

    for (int i = 0; i < 12; ++i) {
        ivec2 edge = SDF_CELL_EDGES[i];
        float d0 = sdf[edge.x];
        float d1 = sdf[edge.y];
        // Sentinels: INFINITY marks corners with no SDF data. A crossing
        // against a sentinel corner yields t ~ 1.0 (vertex parked at the
        // corner) with a wild plane normal, so both are skipped.
        if (!sdfIsFinite(d0) || !sdfIsFinite(d1)) {
            continue;
        }
        if (abs(d0) > SDF_INF * 0.5f || abs(d1) > SDF_INF * 0.5f) {
            continue;
        }
        bool sign0 = d0 < 0.0;
        bool sign1 = d1 < 0.0;
        if (sign0 != sign1) {
            vec3 p0 = sdfCellCorner(mn, sz, edge.x);
            vec3 p1 = sdfCellCorner(mn, sz, edge.y);
            float denom = d0 - d1;
            precise float t = (denom != 0.0) ? clamp(d0 / denom, 0.0, 1.0) : 0.5;
            precise vec3 p = p0 * (1.0 - t) + p1 * t;
            precise vec3 nm = normals[edge.x] * (1.0 - t) + normals[edge.y] * t;
            precise float nl = sqrt(nm.x * nm.x + nm.y * nm.y + nm.z * nm.z);
            if (!sdfIsFinite(nl) || nl < 1e-6) {
                continue;
            }
            precise vec3 n = nm / nl;
            // glm::outerProduct(n, n) and glm::dot(n, p) written out: builtins
            // are only accurate to a few ulp, which the near-singular QEF below
            // amplifies into visible position drift.
            ATA += mat3(n * n.x, n * n.y, n * n.z);
            ATb += n * (n.x * p.x + n.y * p.y + n.z * p.z);
            massPoint += p;
            count++;
        }
    }

    if (count == 0) {
        // No usable QEF crossing (only diagonal corners straddle and every edge
        // pair is blocked by a sentinel): average the finite straddling edges so
        // the vertex sits ON the real surface segment.
        vec3 fallback = vec3(0.0);
        int n = 0;
        for (int i = 0; i < 12; ++i) {
            ivec2 edge = SDF_CELL_EDGES[i];
            float d0 = sdf[edge.x];
            float d1 = sdf[edge.y];
            if (!sdfIsFinite(d0) || !sdfIsFinite(d1)) continue;
            bool sign0 = d0 < 0.0;
            bool sign1 = d1 < 0.0;
            if (sign0 == sign1) continue;
            float denom = d0 - d1;
            precise float t = (denom != 0.0) ? clamp(d0 / denom, 0.0, 1.0) : 0.5;
            fallback += sdfCellCorner(mn, sz, edge.x) * (1.0 - t) + sdfCellCorner(mn, sz, edge.y) * t;
            ++n;
        }
        if (n > 0) {
            return fallback / float(n);
        }
        return mn + vec3(sz * 0.5);
    }

    massPoint /= float(count);

    // Regularize toward the mass point so an underconstrained QEF does not
    // spike (same 1e-2 factor as the CPU).
    ATA[0][0] += 1e-2;
    ATA[1][1] += 1e-2;
    ATA[2][2] += 1e-2;
    ATb += massPoint * 1e-2;

    precise vec3 result = sdfSolveLinearSystem(ATA, ATb);

    if (any(isnan(result)) || any(isinf(result))) {
        return massPoint;
    }
    if (any(isnan(massPoint)) || any(isinf(massPoint))) {
        return mn + vec3(sz * 0.5);
    }
    precise vec3 clamped = clamp(result, mn, mn + vec3(sz));
    return clamped;
}

#endif // SDF_CELL_GLSL

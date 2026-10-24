#include "SDF.hpp"
#include "../math/Math.hpp"
#include <cmath>
#include <cstring>

glm::vec3 SDF::getPosition(float sdf[8], const BoundingCube &cube) {
    SpaceType eval = SDF::eval(sdf);
    if(eval != SpaceType::Surface) {
        return cube.getCenter();
    }

    // Exact-zero corner: the surface passes through a cell corner (e.g. a
    // shape exactly tangent to the cell, like a sphere rim touching a shared
    // corner). Neighboring cells then solve slightly different QEF systems
    // (different normals/crossings), so their near-corner vertices differ in
    // the low bits and the shared edges split into two n=1 edges — a crack.
    // Snap onto the corner instead: every cell sharing the corner stores the
    // same bit-exact zero, so all emit the identical position.
    for(int i = 0; i < 8; ++i) {
        if(sdf[i] == 0.0f) {
            return cube.getCorner(i);
        }
    }

    glm::vec3 normals[8];
    for (int i = 0; i < 8; ++i) {
        normals[i] = SDF::getNormalFromPosition(sdf, cube, cube.getCorner(i));
    }

    glm::mat3 ATA(0.0f);
    glm::vec3 ATb(0.0f);
    glm::vec3 massPoint(0.0f);
    int count = 0;

    for (int i = 0; i < 12; ++i) {
        glm::ivec2 edge = SDF_EDGES[i];
        float d0 = sdf[edge[0]];
        float d1 = sdf[edge[1]];

        // Sentinels: INFINITY marks corners with no SDF data (untouched /
        // pruned / outside the domain). A crossing against a sentinel corner
        // yields t ≈ 1.0, parking the contour vertex at the corner, and its
        // plane normal is wild — both corrupt the QEF/massPoint. Only
        // interpolate across corners that carry a real, finite sample.
        if (!std::isfinite(d0) || !std::isfinite(d1)) {
            continue;
        }
        if (std::fabs(d0) > INFINITY * 0.5f || std::fabs(d1) > INFINITY * 0.5f) {
            continue;
        }

        bool sign0 = d0 < 0.0f;
        bool sign1 = d1 < 0.0f;

        if (sign0 != sign1) {
            glm::vec3 p0 = cube.getCorner(edge[0]);
            glm::vec3 p1 = cube.getCorner(edge[1]);
            float denom = d0 - d1;
            float t = (denom != 0.0f) ? glm::clamp(d0 / denom, 0.0f, 1.0f) : 0.5f;
            glm::vec3 p = glm::mix(p0, p1, t);
            glm::vec3 nm = glm::mix(normals[edge[0]], normals[edge[1]], t);
            float nl = glm::length(nm);
            if (!std::isfinite(nl) || nl < 1e-6f) {
                continue;
            }
            glm::vec3 n = nm / nl;

            ATA += glm::outerProduct(n, n);
            ATb += n * glm::dot(n, p);

            massPoint += p;
            count++;
        }
    }

    if (count == 0) {
        // No usable QEF crossing: this happens when the only straddling
        // corners are diagonal and every edge pair is blocked by a sentinel
        // (INFINITY) corner. Parking the vertex at the cell center would
        // float garbage triangles in mid-air/buried ground, so instead
        // average the crossings of the finite straddling edges — the vertex
        // then sits ON the real surface segment.
        glm::vec3 fallback(0.0f);
        int n = 0;
        for (int i = 0; i < 12; ++i) {
            glm::ivec2 edge = SDF_EDGES[i];
            float d0 = sdf[edge[0]];
            float d1 = sdf[edge[1]];
            if (!std::isfinite(d0) || !std::isfinite(d1)) continue;
            bool sign0 = d0 < 0.0f;
            bool sign1 = d1 < 0.0f;
            if (sign0 == sign1) continue;
            float denom = d0 - d1;
            float t = (denom != 0.0f) ? glm::clamp(d0 / denom, 0.0f, 1.0f) : 0.5f;
            fallback += glm::mix(cube.getCorner(edge[0]), cube.getCorner(edge[1]), t);
            ++n;
        }
        if (n > 0) {
            return fallback / (float)n;
        }
        return cube.getCenter();
    }

    massPoint /= (float)count;

    // Regularize toward the mass point to prevent spikes when
    // the QEF is underconstrained (few edge crossings or near-parallel normals)
    ATA[0][0] += 1e-2f;
    ATA[1][1] += 1e-2f;
    ATA[2][2] += 1e-2f;
    ATb += massPoint * 1e-2f;

    glm::vec3 result = Math::solveLinearSystem(ATA, ATb);

    if (glm::any(glm::isnan(result)) || glm::any(glm::isinf(result))) {
        return massPoint;
    }
    if (glm::any(glm::isnan(massPoint)) || glm::any(glm::isinf(massPoint))) {
        return cube.getCenter();
    }

    glm::vec3 min = cube.getMin();
    glm::vec3 max = cube.getMax();
    result = glm::clamp(result, min, max);

    return result;
}

glm::vec3 SDF::getNormalFromPosition(float sdf[8], const BoundingCube& cube, const glm::vec3& position) {
    glm::vec3 local = (position - cube.getMin()) / cube.getLength(); // Convert to [0,1]^3 within cube

    // Trilinear interpolation gradient
    // CUBE_CORNERS convention: bit2=x, bit1=y, bit0=z
    // dx: ∂f/∂x — x-differences (bit2) weighted by (y,z)
    float dx = (
        (1 - local.y) * (1 - local.z) * (sdf[4] - sdf[0]) +
        (1 - local.y) * local.z * (sdf[5] - sdf[1]) +
        local.y * (1 - local.z) * (sdf[6] - sdf[2]) +
        local.y * local.z * (sdf[7] - sdf[3])
    );

    // dy: ∂f/∂y — y-differences (bit1) weighted by (x,z)
    float dy = (
        (1 - local.x) * (1 - local.z) * (sdf[2] - sdf[0]) +
        local.x * (1 - local.z) * (sdf[3] - sdf[1]) +
        (1 - local.x) * local.z * (sdf[6] - sdf[4]) +
        local.x * local.z * (sdf[7] - sdf[5])
    );

    // dz: ∂f/∂z — z-differences (bit0) weighted by (x,y)
    float dz = (
        (1 - local.x) * (1 - local.y) * (sdf[1] - sdf[0]) +
        (1 - local.x) * local.y * (sdf[3] - sdf[2]) +
        local.x * (1 - local.y) * (sdf[5] - sdf[4]) +
        local.x * local.y * (sdf[7] - sdf[6])
    );

    glm::vec3 normal = glm::vec3(dx, dy, dz) / cube.getLength();
    float nl = glm::length(normal);
    if (!std::isfinite(nl) || nl < 1e-6f) {
        return glm::vec3(0.0f, 1.0f, 0.0f);
    }
    return normal / nl;
}

// Safe trilinear lerp for SDF fields that may carry INFINITY sentinels (corners
// with no data / outside the known domain). glm::mix(a, b, t) evaluates
// a*(1-t) + b*t, which is NaN when one operand is INFINITY and the other factor
// is exactly 0 (e.g. t==0 with b==INFINITY: 0*INFINITY == NaN). That NaN
// silently defeats the Simplifier's re-interpolation error check and corrupts
// vertex placement / cell-to-cell mesh stitching. We guard the boundary t values
// and let an infinite endpoint dominate for any interior t — correct for a union
// SDF, whose field grows without bound toward the unknown region.
static inline float lerpSDF(float a, float b, float t) {
    if (t <= 0.0f) return a;
    if (t >= 1.0f) return b;
    if (a == INFINITY || b == INFINITY) return INFINITY;
    return a + (b - a) * t;
}

float SDF::interpolate(const float sdf[8], const glm::vec3 &position, const BoundingCube &cube) {
    glm::vec3 local = (position - cube.getMin()) / cube.getLength(); // [0,1]^3
    float x = local.x, y = local.y, z = local.z;

    if(x== 0.0f) {
        if(y == 0.0f) {
            if(z == 0.0f) return sdf[0];
            if(z == 1.0f) return sdf[1];
        }
        if(y == 1.0f) {
            if(z == 0.0f) return sdf[2];
            if(z == 1.0f) return sdf[3];
        }
    } else if(x == 1.0f) {
        if(y == 0.0f) {
            if(z == 0.0f) return sdf[4];
            if(z == 1.0f) return sdf[5];
        }
        if(y == 1.0f) {
            if(z == 0.0f) return sdf[6];
            if(z == 1.0f) return sdf[7];
        }
    }
    // Interpolate along z for each (x, y) pair
    float v000 = lerpSDF(sdf[0], sdf[1], z); // (0,0,0)-(0,0,1)
    float v010 = lerpSDF(sdf[2], sdf[3], z); // (0,1,0)-(0,1,1)
    float v100 = lerpSDF(sdf[4], sdf[5], z); // (1,0,0)-(1,0,1)
    float v110 = lerpSDF(sdf[6], sdf[7], z); // (1,1,0)-(1,1,1)

    // Interpolate along y
    float v00 = lerpSDF(v000, v010, y); // (0,*,*)
    float v10 = lerpSDF(v100, v110, y); // (1,*,*)

    // Interpolate along x
    return lerpSDF(v00, v10, x);
}

void SDF::getChildSDF(const float sdf[8], unsigned int i, float result[8]) {
    BoundingCube canonicalCube = BoundingCube(glm::vec3(0.0f), 1.0f);
    BoundingCube cube = canonicalCube.getChild(i);
    // Interpolate each child corner from the parent field instead of flood-filling
    // the whole child to INFINITY when any parent corner is a sentinel. A single
    // unknown corner must only poison the child corners that actually depend on it;
    // the rest of the child keeps the correctly interpolated (finite) field, so
    // shape() can descend and materialize real geometry instead of dropping it.
    for (unsigned int j = 0; j < 8; ++j) {
        glm::vec3 corner = cube.getCorner(j);
        result[j] = interpolate(sdf, corner, canonicalCube);
    }
}

void SDF::copySDF(const float src[8], float dst[8]) {
    memcpy(dst, src, sizeof(float)*8);
}

SpaceType SDF::eval(const float sdf[8]) {
    bool hasPositive = false;
    bool hasNegative = false;
    for (int i = 0; i < 8; ++i) {
        if(sdf[i] == INFINITY) {
            continue;  // skip untouched corners
        } else if (sdf[i] >= 0.0f) {
            hasPositive = true;
        } else {
            hasNegative = true;
        }
    }
    if (!hasPositive && !hasNegative) return SpaceType::Empty;
    return hasNegative && hasPositive ? SpaceType::Surface : (hasPositive ? SpaceType::Empty : SpaceType::Solid);
}

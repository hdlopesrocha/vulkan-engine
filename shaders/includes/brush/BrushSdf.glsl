#ifndef BRUSH_SDF_GLSL
#define BRUSH_SDF_GLSL

#include "../noise/Perlin.glsl"
#include "../sdf/SdfNoise.glsl"

// ── Brush SDF (set=1 binding=0) ────────────────────────────────────────────
// Shared by the brush preview raymarcher (BrushSdf.frag) and the solid
// shader's PAINT/REMOVE intersection test (SolidSurface.glsl). The field is a
// GLSL mirror of the CPU brush stack (sdf/*DistanceFunction + *DistanceEffect
// + SweepSignedDistanceFunction), with these documented approximations:
//   * Perlin effects use the engine's classic gradient noise (Perlin.glsl)
//     instead of stb_perlin; the fields are visually equivalent but not
//     bit-identical.
//   * The VoronoiCarve cell offsets use integer hash noise (sdfHashU) instead
//     of perlin-at-lattice; same cell structure, deterministic and cheap.
// Layout must stay in sync with vulkan/ubo/BrushSdfUBO.hpp.
layout(std140, set = 1, binding = 0) uniform BrushSdfBlock {
    vec4  xTranslate;
    vec4  xScale;
    vec4  xRotation;     // quaternion xyzw
    vec4  xParams0;
    vec4  xParams1;
    vec4  xEffect0;      // amplitude, frequency, threshold, cellSize
    vec4  xEffect1;      // brightness, contrast, unused, unused
    vec4  xSweepStart;   // xyz = previous world translate
    vec4  xBounds;       // xyz = world bounding sphere center, w = radius
    uvec4 xFlags;        // x=sdfType, y=effectType+1 (0=off), z=material, w=sweep
    vec4  xHsv;          // xyz = paint tint HSV
    vec4  xViewport;     // xy = target size in pixels
} brush;

// ── Transform helpers (CPU parity: inverse(quat) * (p - translate)) ─────────
vec3 brushRotate(vec3 v, vec4 q) {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
vec3 brushRotateInv(vec3 v, vec4 q) {
    return brushRotate(v, vec4(-q.xyz, q.w));
}

// ── Primitives (exact mirrors of sdf/SDF.cpp) ───────────────────────────────
float brushSdBox(vec3 p, vec3 halfExtents) {
    vec3 q = abs(p) - halfExtents;
    return length(max(q, vec3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float brushSdCapsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-9), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

float brushSdTaperedCapsule(vec3 p, vec3 a, vec3 b, float r1, float r2) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-9), 0.0, 1.0);
    float r = mix(r1, r2, h);
    return length(pa - ba * h) - r;
}

float brushSdTorus(vec3 p, vec2 t) {
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}

float brushSdCylinder(vec3 p, float r, float h) {
    vec2 d = vec2(length(p.xz) - r, abs(p.y) - h);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2(0.0)));
}

float brushSdTaperedCylinder(vec3 p, float r1, float r2, float h) {
    vec2 q = vec2(length(p.xz), p.y);
    vec2 k1 = vec2(r2, h);
    vec2 k2 = vec2(r2 - r1, 2.0 * h);
    vec2 ca = vec2(q.x - min(q.x, (q.y < 0.0) ? r1 : r2), abs(q.y) - h);
    vec2 cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / max(dot(k2, k2), 1e-9), 0.0, 1.0);
    float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
    return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}

float brushSdOctahedron(vec3 p, float s) {
    p = abs(p);
    float m = p.x + p.y + p.z - s;
    vec3 q;
    if (3.0 * p.x < m)      q = p.xyz;
    else if (3.0 * p.y < m) q = p.yzx;
    else if (3.0 * p.z < m) q = p.zxy;
    else                    return m * 0.57735027;
    float k = clamp(0.5 * (q.z - q.y + s), 0.0, s);
    return length(vec3(q.x, q.y - s + k, q.z - k));
}

// CPU SDF::cone: unit cone, apex at origin, base radius = height = 1.
float brushSdConeUnit(vec3 p) {
    const vec2 q = vec2(1.0, -1.0);
    vec2 w = vec2(length(p.xz), p.y);
    vec2 a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0);
    vec2 b = w - q * vec2(clamp(w.x / q.x, 0.0, 1.0), 1.0);
    float k = sign(q.y);
    float d = min(dot(a, a), dot(b, b));
    float s = max(k * (w.x * q.y - w.y * q.x), k * (w.y - q.y));
    return sqrt(d) * sign(s);
}

float brushSdSegment(vec3 p, vec3 a, vec3 b) {
    vec3 pa = p - a, ba = b - a;
    float t = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-9), 0.0, 1.0);
    return length(pa - ba * t);
}

float brushSdTriangle(vec3 p, vec3 a, vec3 b, vec3 c) {
    vec3 ab = b - a, ac = c - a;
    vec3 n = cross(ab, ac);
    float nlen2 = dot(n, n);
    if (nlen2 < 1e-12) {
        return min(brushSdSegment(p, a, b), min(brushSdSegment(p, b, c), brushSdSegment(p, c, a)));
    }
    vec3 nn = n / sqrt(nlen2);
    float distPlane = dot(p - a, nn);
    vec3 proj = p - distPlane * nn;
    vec3 ap = proj - a, bp = proj - b, cp = proj - c;
    if (dot(cross(ab, ap), n) >= 0.0 &&
        dot(cross(c - b, bp), n) >= 0.0 &&
        dot(cross(a - c, cp), n) >= 0.0) {
        return abs(distPlane);
    }
    return min(brushSdSegment(p, a, b), min(brushSdSegment(p, b, c), brushSdSegment(p, c, a)));
}

// CPU SDF::pyramid: square base, apex at (0, h, 0), signed by inside test.
float brushSdPyramid(vec3 p, float h, float a) {
    vec3 apex = vec3(0.0, h, 0.0);
    vec3 v0 = vec3(-a, 0.0, -a);
    vec3 v1 = vec3( a, 0.0, -a);
    vec3 v2 = vec3( a, 0.0,  a);
    vec3 v3 = vec3(-a, 0.0,  a);
    vec3 geoCenter = (apex + v0 + v1 + v2 + v3) / 5.0;

    float d0 = brushSdTriangle(p, apex, v0, v1);
    float d1 = brushSdTriangle(p, apex, v1, v2);
    float d2 = brushSdTriangle(p, apex, v2, v3);
    float d3 = brushSdTriangle(p, apex, v3, v0);
    float db0 = brushSdTriangle(p, v0, v1, v2);
    float db1 = brushSdTriangle(p, v2, v3, v0);
    float dist = min(min(min(d0, d1), min(d2, d3)), min(db0, db1));

    vec3 n0 = normalize(cross(v0 - apex, v1 - apex));
    if (dot(n0, geoCenter - apex) > 0.0) n0 = -n0;
    vec3 n1 = normalize(cross(v1 - apex, v2 - apex));
    if (dot(n1, geoCenter - apex) > 0.0) n1 = -n1;
    vec3 n2 = normalize(cross(v2 - apex, v3 - apex));
    if (dot(n2, geoCenter - apex) > 0.0) n2 = -n2;
    vec3 n3 = normalize(cross(v3 - apex, v0 - apex));
    if (dot(n3, geoCenter - apex) > 0.0) n3 = -n3;
    vec3 nb0 = normalize(cross(v1 - v0, v2 - v0));
    if (dot(nb0, geoCenter - v0) > 0.0) nb0 = -nb0;
    vec3 nb1 = normalize(cross(v3 - v2, v0 - v2));
    if (dot(nb1, geoCenter - v2) > 0.0) nb1 = -nb1;

    bool inside =
        (dot(p - apex, n0) <= 0.0) && (dot(p - apex, n1) <= 0.0) &&
        (dot(p - apex, n2) <= 0.0) && (dot(p - apex, n3) <= 0.0) &&
        (dot(p - v0, nb0) <= 0.0) && (dot(p - v2, nb1) <= 0.0);
    return inside ? -dist : dist;
}

// ── Primitive dispatch (sdfType → local-space field) ────────────────────────
// `xlate` is the primitive's own world translate so the sweep path can
// evaluate the START placement with the same routine.
float brushPrimitiveAt(vec3 p, vec3 xlate, vec4 params0, vec4 params1) {
    vec3 pos = brushRotateInv(p - xlate, brush.xRotation);
    vec3 s = max(abs(brush.xScale.xyz), vec3(1e-6));
    float minS = min(min(s.x, s.y), s.z);
    uint type = brush.xFlags.x;
    if (type == 0u) {          // Sphere (scale = radii)
        vec3 q = abs(pos) / s;
        return (length(q) - 1.0) * minS;
    } else if (type == 1u) {   // Box (scale = half extents)
        return brushSdBox(pos, s);
    } else if (type == 2u) {   // Capsule
        return brushSdCapsule(pos / s, params0.xyz, params1.xyz, params0.w) * minS;
    } else if (type == 3u) {   // Octahedron
        return brushSdOctahedron(pos / s, 1.0) * minS;
    } else if (type == 4u) {   // Pyramid
        return brushSdPyramid(pos / s, 1.0, sqrt(0.5)) * minS;
    } else if (type == 5u) {   // Torus
        return brushSdTorus(pos / s, params0.xy) * minS;
    } else if (type == 6u) {   // Cone
        return brushSdConeUnit(pos / s - vec3(0.0, 1.0, 0.0)) * minS;
    } else if (type == 7u) {   // Cylinder
        return brushSdCylinder(pos / s, 0.5, 1.0) * minS;
    } else if (type == 8u) {   // Tapered cylinder
        return brushSdTaperedCylinder(pos / s, params0.x, params0.y, 0.5) * minS;
    }                          // 9 = Tapered capsule
    return brushSdTaperedCapsule(pos / s, params0.xyz, params1.xyz, params0.w, params1.w) * minS;
}

// Sweep trail (CPU SweepSignedDistanceFunction): the start/end transforms
// share scale and rotation, so the inverse-mix collapses to evaluating the
// START placement at the point offset from the start center.
float brushRawDistance(vec3 p) {
    if (brush.xFlags.w == 0u) {
        return brushPrimitiveAt(p, brush.xTranslate.xyz, brush.xParams0, brush.xParams1);
    }
    vec3 posA = brush.xSweepStart.xyz;
    vec3 posB = brush.xTranslate.xyz;
    vec3 seg = posB - posA;
    float segLenSq = dot(seg, seg);
    if (segLenSq < 1e-8) {
        return brushPrimitiveAt(p, posB, brush.xParams0, brush.xParams1);
    }
    float t = clamp(dot(p - posA, seg) / segLenSq, 0.0, 1.0);
    vec3 closest = posA + t * seg;
    vec3 q = posA + brushRotate(brushRotateInv(p - closest, brush.xRotation), brush.xRotation);
    return brushPrimitiveAt(q, posA, brush.xParams0, brush.xParams1);
}

// ── Effects (CPU parity, see approximation note at the top) ─────────────────
float brushBc(float c, float brightness, float contrast) {
    c = clamp(c + brightness, -1.0, 1.0);
    return clamp(c * contrast, -1.0, 1.0);
}

vec3 brushPerlinFractal(vec3 p, float frequency) {
    vec3 total = vec3(0.0);
    float freq = frequency;
    float amp = 1.0;
    for (int i = 0; i < 6; ++i) {
        total += amp * vec3(
            perlinNoise3D(p * freq + vec3(0.0)),
            perlinNoise3D(p * freq + vec3(100.0)),
            perlinNoise3D(p * freq + vec3(200.0)));
        freq *= 2.0;
        amp *= 0.5;
    }
    return total;
}

float brushCarveFractal(vec3 p, float threshold, float frequency) {
    float noiseValue = 0.0;
    float freq = frequency;
    float amp = 1.0;
    for (int i = 0; i < 6; ++i) {
        noiseValue += amp * perlinNoise3D(p * freq);
        freq *= 2.0;
        amp *= 0.5;
    }
    if (noiseValue > threshold) return noiseValue - threshold;
    return 0.0;
}

float brushCellRand(ivec3 cell, uint channel) {
    uvec3 c = uvec3(cell) + uvec3(13u, 29u, 43u) * (channel + 1u);
    return float(sdfHashU(c)) * (1.0 / 4294967295.0);
}

float brushVoronoi3D(vec3 p, float cellSize) {
    if (cellSize <= 0.0) return 0.0;
    vec3 q = p / cellSize;
    q += 0.3 * vec3(perlinNoise3D(q), perlinNoise3D(q.yzx), perlinNoise3D(q.zxy));
    ivec3 baseCell = ivec3(floor(q));
    float minDist = 1e10;
    for (int k = -1; k <= 1; ++k) {
        for (int j = -1; j <= 1; ++j) {
            for (int i = -1; i <= 1; ++i) {
                ivec3 neighbor = baseCell + ivec3(i, j, k);
                vec3 cellSeed = vec3(neighbor) + vec3(
                    brushCellRand(neighbor, 0u),
                    brushCellRand(neighbor, 1u),
                    brushCellRand(neighbor, 2u));
                minDist = min(minDist, length(q - cellSeed));
            }
        }
    }
    return minDist / sqrt(3.0);
}

// ── Public field evaluation ─────────────────────────────────────────────────
// World-space signed distance of the brush volume. Effects wrap the primitive
// (or its sweep) exactly like applyBrushWithEffect does on the CPU.
float brushDistance(vec3 p) {
    uint effect = brush.xFlags.y;
    if (effect == 0u) return brushRawDistance(p);

    vec3 translate = brush.xTranslate.xyz;
    vec3 localP = p - translate;
    float amp = brush.xEffect0.x;
    float freq = max(brush.xEffect0.y, 1e-6);
    float brightness = brush.xEffect1.x;
    float contrast = brush.xEffect1.y;

    if (effect == 1u) {  // PerlinDistort
        vec3 n = brushPerlinFractal(localP, freq);
        n = vec3(brushBc(n.x, brightness, contrast),
                 brushBc(n.y, brightness, contrast),
                 brushBc(n.z, brightness, contrast));
        vec3 np = localP + amp * n;
        return brushRawDistance(np + translate) / (1.0 + 18.0 * amp * freq);
    } else if (effect == 2u) {  // PerlinCarve
        float d = brushRawDistance(p);
        float noise = brushBc(brushCarveFractal(localP, brush.xEffect0.z, freq), brightness, contrast);
        return (d + noise * amp) / (1.0 + amp * freq * 6.0);
    } else if (effect == 3u) {  // SineDistort
        float dx = sin(localP.x * freq) * cos(localP.y * freq) * sin(localP.z * freq);
        float dy = cos(localP.x * freq) * sin(localP.y * freq) * cos(localP.z * freq);
        float dz = sin(localP.x * freq) * sin(localP.y * freq) * cos(localP.z * freq);
        vec3 np = localP + (amp * 0.5) * vec3(dx, dy, dz);
        return brushRawDistance(np + translate) / (1.0 + 1.5 * amp * freq);
    }
    // VoronoiCarve
    float d = brushRawDistance(p);
    float noise = brushBc(brushVoronoi3D(localP, brush.xEffect0.w), brightness, contrast);
    return (d - amp * noise) / (1.0 + 2.0 * amp / max(brush.xEffect0.w, 1e-6));
}

// Bounding-sphere early-out shared by the preview, the solid PAINT/REMOVE
// test and the gradient taps.
bool brushInsideBounds(vec3 p) {
    float r = brush.xBounds.w;
    if (r <= 0.0) return false;
    vec3 d = p - brush.xBounds.xyz;
    return dot(d, d) <= r * r;
}

#endif  // BRUSH_SDF_GLSL

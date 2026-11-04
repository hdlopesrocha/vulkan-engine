#ifndef OCTREE_RAYMARCH_TABLE_GLSL
#define OCTREE_RAYMARCH_TABLE_GLSL

// GPU mirror of utils/OctreeRaymarchTable.hpp.
//
// One flat node array per frame slot (set=2 binding=0) plus the per-layer root
// indices (binding=1). The table stores surface octree nodes only; the parent's
// meta.w packs the information the descent needs for absent children:
//   bit0       leaf
//   bits8..15  surface children pruned by the node cap (per corner slot)
//   bits16..31 two-bit SpaceType per child slot (Empty / Surface / Solid)
//
// Keep the struct order/size identical to OctreeRaymarchNode (128 bytes).
//
// std430 is sufficient here: every member is a vec4/ivec4, so the scalar and
// std430 layouts are byte-identical and the shader does not depend on the
// optional scalarBlockLayout device feature.

struct RaymarchNode {
    vec4  sdf0;        // sdf[0..3]
    vec4  sdf1;        // sdf[4..7]
    vec4  cubeMinSize; // xyz = node cube min, w = edge length
    ivec4 meta;        // x = lod, y = SpaceType, z = brushIndex, w = flags
    ivec4 child03;     // child index per corner slot 0..3 (-1 = absent)
    ivec4 child47;     // child index per corner slot 4..7 (-1 = absent)
    vec4  hsv0;        // hsv.xyz, textureType
    vec4  hsv1;        // data.xy, 0, 0
};

layout(std430, set = 2, binding = 0) readonly buffer RaymarchNodes {
    RaymarchNode rmNodes[];
};
layout(std430, set = 2, binding = 1) readonly buffer RaymarchRoots {
    uint rmRoots[];
};

const int RM_TYPE_EMPTY = 0;
const int RM_TYPE_SURFACE = 1;
const int RM_TYPE_SOLID = 2;

// Positive infinity sentinel used by the stored corner samples: a corner with
// no SDF data is skipped by the interpolation, exactly like OctreeTriangles.comp.
#define RM_INF (uintBitsToFloat(0x7F800000u))

int rmNodeChild(uint id, int slot) {
    ivec4 c = (slot < 4) ? rmNodes[id].child03 : rmNodes[id].child47;
    return c[slot & 3];
}

bool rmNodeIsLeaf(uint id) { return (rmNodes[id].meta.w & 1) != 0; }

bool rmNodeChildPruned(uint id, int slot) {
    return ((rmNodes[id].meta.w >> (8 + slot)) & 1) != 0;
}

// Non-surface leaf type for an absent child slot. Slots with no child at all
// read as Empty (0), which is also the safe interpretation for a skipped region.
int rmNodeChildType(uint id, int slot) {
    return (rmNodes[id].meta.w >> (16 + slot * 2)) & 3;
}

vec3 rmNodeMin(uint id) { return rmNodes[id].cubeMinSize.xyz; }
float rmNodeSize(uint id) { return rmNodes[id].cubeMinSize.w; }

float rmNodeSdf(uint id, int idx) {
    vec4 v = (idx < 4) ? rmNodes[id].sdf0 : rmNodes[id].sdf1;
    return v[idx & 3];
}

bool rmIsFinite(float v) { return !(isnan(v) || isinf(v)); }

// Trilinear field over one node's eight stored corner samples.
//
// INFINITY marks a corner with no SDF data (untouched / pruned / outside the
// domain), exactly like OctreeTriangles.comp. The CPU SDF::interpolate()
// propagates that sentinel, which is right for a cell rejection test but fatal
// for sphere tracing: a surface cell with one unknown corner would evaluate to
// INF over a large part of the cell and the ray would miss the isosurface that
// really lives there. Instead, evaluate over the finite corners only and
// renormalize their trilinear weights. All-finite cells are unchanged (the
// weights already sum to 1); cells dominated by unknown corners return INF
// (no usable field) and the descent continues in a sibling / parent.
float rmTrilinear(uint id, vec3 p) {
    vec3 mn = rmNodeMin(id);
    float sz = rmNodeSize(id);
    vec3 l = clamp((p - mn) / max(sz, 1e-8), 0.0, 1.0);

    float wx0 = 1.0 - l.x, wy0 = 1.0 - l.y, wz0 = 1.0 - l.z;
    float w[8];
    w[0] = wx0 * wy0 * wz0;
    w[1] = wx0 * wy0 * l.z;
    w[2] = wx0 * l.y * wz0;
    w[3] = wx0 * l.y * l.z;
    w[4] = l.x * wy0 * wz0;
    w[5] = l.x * wy0 * l.z;
    w[6] = l.x * l.y * wz0;
    w[7] = l.x * l.y * l.z;

    float acc = 0.0;
    float wsum = 0.0;
    for (int i = 0; i < 8; ++i) {
        float v = rmNodeSdf(id, i);
        if (!rmIsFinite(v)) continue;
        acc += w[i] * v;
        wsum += w[i];
    }
    if (wsum <= 1e-8) return RM_INF;
    return acc / wsum;
}

// Guarded reciprocal: a component of exactly zero must not turn a slab
// difference of exactly zero into 0 * inf = NaN.
float rmInvComp(float v) {
    if (abs(v) >= 1e-9) return 1.0 / v;
    return (v < 0.0) ? -1e9 : 1e9;
}

// Ray/AABB entry/exit in parametric units. Returns false when the ray misses
// the box or the box lies entirely behind the origin.
bool rmRayBox(vec3 boxMin, float boxSize, vec3 ro, vec3 rd, out float t0, out float t1) {
    vec3 boxMax = boxMin + vec3(boxSize);
    vec3 inv = vec3(rmInvComp(rd.x), rmInvComp(rd.y), rmInvComp(rd.z));
    vec3 a = (boxMin - ro) * inv;
    vec3 b = (boxMax - ro) * inv;
    vec3 lo = min(a, b);
    vec3 hi = max(a, b);
    t0 = max(max(lo.x, lo.y), lo.z);
    t1 = min(min(hi.x, hi.y), hi.z);
    return t1 >= max(t0, 0.0);
}

#endif // OCTREE_RAYMARCH_TABLE_GLSL

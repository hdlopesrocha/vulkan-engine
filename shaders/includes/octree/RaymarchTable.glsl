#ifndef OCTREE_RAYMARCH_TABLE_GLSL
#define OCTREE_RAYMARCH_TABLE_GLSL

// GPU mirror of utils/OctreeRaymarchTable.hpp.
//
// One flat node array per frame slot (set=2 binding=0), the per-layer root
// entries (binding=1: node index + stable Layer::Id) and the derived QEF
// payload (binding=2, hit-only). The table stores surface
// octree nodes only; the parent's meta.w packs the information the descent
// needs for absent children (see RaymarchNodeRecord.glsl for the flag layout).
//
// The record itself lives in RaymarchNodeRecord.glsl so the once-per-revision
// payload derive pass (OctreeDerive.comp) declares the same 96-byte layout.
// std430 is sufficient here: the node record consists only of vec4/ivec4 and
// the root record only of 4-byte scalars followed by two vec4s, so the scalar
// and std430 layouts are byte-identical and the shader does not depend on the
// optional scalarBlockLayout device feature.

#include "RaymarchNodeRecord.glsl"
#include "RaymarchPayload.glsl"

// Root entry: node index + stable Layer::Id + renderer kind (0 = Solid,
// 1 = Water) + the layer's conservative surface AABB. The AABB bounds every
// surface point the root march can sample (the union of the stored surface
// cells, not the whole layer cube), so a missed AABB can be skipped and a hit
// can clamp every later root without eliding geometry. std430 size is 48 B
// (4 x 4 B header + 2 x vec4). Must match OctreeRaymarchRoot.
struct RaymarchRoot {
    uint node;
    int  layerId;
    uint kind;
    uint pad;
    vec4 surfaceMin; // xyz = surface AABB min, w unused
    vec4 surfaceMax; // xyz = surface AABB max, w unused
};

layout(std430, set = 2, binding = 0) readonly buffer RaymarchNodes {
    RaymarchNode rmNodes[];
};
layout(std430, set = 2, binding = 1) readonly buffer RaymarchRoots {
    RaymarchRoot rmRoots[];
};
// Once-per-revision derived QEF position/normal. Only hit-time ring lookups
// read it, so it stays a separate hit-only stream instead of growing the
// traversal record.
layout(std430, set = 2, binding = 2) readonly buffer RaymarchPayloads {
    RaymarchPayload rmPayload[];
};
const int RM_TYPE_EMPTY = 0;
const int RM_TYPE_SURFACE = 1;
const int RM_TYPE_SOLID = 2;

// Positive infinity sentinel used by the stored corner samples: a corner with
// no SDF data is skipped by the interpolation, exactly like OctreeTriangles.comp.
#define RM_INF (uintBitsToFloat(0x7F800000u))

// Compact child header: stored Surface children are contiguous from
// meta.y (childBase), hsv1.w is the 8-bit presence mask. Slot s resolves to
// childBase + bitCount(mask & ((1 << s) - 1)), which is the popcount of the
// present slots below s. bitCount() is core GLSL 4.0.
int rmNodeChild(uint id, int slot) {
    uint mask = uint(rmNodes[id].hsv1.w + 0.5);
    if ((mask & (1u << slot)) == 0u) return -1;
    uint base = uint(rmNodes[id].meta.y);
    return int(base + bitCount(mask & ((1u << slot) - 1u)));
}

bool rmNodeIsLeaf(uint id) { return (rmNodes[id].meta.w & int(RM_FLAG_LEAF)) != 0; }

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
// the box or the box lies entirely behind the origin. `inv` is the ray
// direction reciprocal, hoisted by callers that test more than one box. The
// vec3-extent overload serves non-cubic boxes (the layer surface AABBs); the
// cube overload keeps every existing call site unchanged.
bool rmRayBoxInv(vec3 boxMin, vec3 boxSize, vec3 ro, vec3 inv,
                 out float t0, out float t1) {
    vec3 boxMax = boxMin + boxSize;
    vec3 a = (boxMin - ro) * inv;
    vec3 b = (boxMax - ro) * inv;
    vec3 lo = min(a, b);
    vec3 hi = max(a, b);
    t0 = max(max(lo.x, lo.y), lo.z);
    t1 = min(min(hi.x, hi.y), hi.z);
    return t1 >= max(t0, 0.0);
}

bool rmRayBoxInv(vec3 boxMin, float boxSize, vec3 ro, vec3 inv,
                 out float t0, out float t1) {
    return rmRayBoxInv(boxMin, vec3(boxSize), ro, inv, t0, t1);
}

#endif // OCTREE_RAYMARCH_TABLE_GLSL

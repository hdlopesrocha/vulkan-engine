#version 460
// RT_ENABLED is defined for the OctreeRaymarchRT.frag.spv variant (hardware
// RT path). Without it the shader uses the procedural-sky fallback with
// identical miss baselines, so non-RT hardware stays validation-clean.
#ifdef RT_ENABLED
#extension GL_EXT_ray_query : require
#endif

// ── Octree SDF raymarcher ───────────────────────────────────────────────────
// Optional solid path (Settings: "Octree SDF Raymarch", mirrored in the Ray
// Marching widget): instead of drawing the tessellated solid chunks, this
// fullscreen pass marches the octree's stored corner SDFs and shades the first
// isosurface hit with the same shadeSolidSurface() routine the raster path
// uses (materials/mapping, CSM shadows, RT reflections/contact shadows).
//
// The node table (utils/OctreeRaymarchTable.hpp) stores SURFACE cells only,
// flattened breadth-first per enabled Solid layer. Each layer contributes one
// root entry covering a cube; roots are marched independently and the nearest
// hit wins. Non-surface children are not stored: their SpaceType is packed in
// the parent so Empty children (all corners outside) and Solid children (all
// corners inside) are both skipped — by construction the unbumped isosurface
// cannot cross either. Only node-cap-pruned Surface children fall back to the
// parent's trilinear field. With tessellation displacement active the skip is
// suspended: the bump (below) can move the surface across a child border, so
// those children are sampled through the parent field to keep the marched
// surface continuous with the displaced raster triangles.
//
// Tessellation parity: the raster TES displaces every tessellated vertex along
// its smooth cell normal by (height - 0.5) * tessHeightScale. The march folds
// that offset into the field (f_bumped = f_base - disp), which moves the
// isosurface by exactly the same amount; both paths therefore show the same
// bumped terrain with the same material height maps/gates.
//
// Brush blending: a hit rebuilds the local Surface-Nets triangle (the four
// quadrant cells around the nearest crossing edge, deduped and owner-rotated
// exactly like OctreeTriangles.comp) and shades with up to three unique brush
// slots blended by barycentric weights — the same compression the raster TCS
// performs. Material transitions across cells are therefore smooth instead of
// snapping to the hit cell's brush.
//
// Surface hits are depth-tested and write gl_FragDepth, so the water pass, the
// composite and the droplet pass keep their existing solid-depth dependency.
// Misses discard (no color, no depth).
//
// Debug view: DEBUG_MODE_OCTREE_SDF_RAYMARCH bypasses shading and paints the
// hit with its owning layer's hue plus the leaf-cell lattice/refinement data,
// so the per-layer octree SDF structure can be inspected directly.

// The node-table interface (set=2 bindings) comes first so the table struct is
// visible before the shading includes below.
#include "../../includes/octree/RaymarchTable.glsl"

#include "../../includes/Locations.glsl"
#include "../../includes/SceneBindings.glsl"

#include "../../includes/water/WaterTint.glsl"
#include "../../includes/DebugModes.glsl"
#include "../../includes/Textures.glsl"
// Set=1 binding 0: brush SDF description. shadeSolidSurface() evaluates it for
// PAINT/REMOVE tinting, exactly like the raster solid shader.
#include "../../includes/brush/BrushSdf.glsl"

#include "../../includes/Common.glsl"
#include "../../includes/solid/TangentBasis.glsl"
#include "../../includes/solid/Triplanar.glsl"
#include "../../includes/solid/Displacement.glsl"
#include "../../includes/shadow/Shadows.glsl"
#include "../../includes/Hsv.glsl"

// Hybrid RT: the raymarched surface is shaded by the SAME shadeSolidSurface()
// body as the raster path, so it must see the same TLAS/params/geometry
// bindings and reflection helpers. RtParams provides the struct/helpers only;
// the bindings are compiled out without RT_ENABLED and the shading then falls
// back to the procedural sky exactly like the non-RT raster variant.
#include "../../includes/rt/RtParams.glsl"
#ifdef RT_ENABLED
#include "../../includes/rt/RtSceneBindings.glsl"
#endif

// Screen-space reflection refinement (set 0, bindings 19/20) required by
// shadeSolidSurface(); the RT-less path leaves the env reflection at sky.
layout(set = 0, binding = 19) uniform sampler2D ssrColorTex;
layout(set = 0, binding = 20) uniform sampler2D ssrDepthTex;

// Set=2: octree node table (bindings declared in RaymarchTable.glsl).

layout(push_constant) uniform OctreeRaymarchPush {
    vec4 viewport;  // xy = solid pass target extent in pixels
    uint rootCount; // number of layer roots in the set-2 root buffer
    uint pad0;
    uint pad1;
    uint pad2;
} pc;

layout(location = FRAG_OUT_COLOR) out vec4 outColor;

// ── Fragment globals consumed by shadeSolidSurface() ───────────────────────
// The raster path fills these from tessellation varyings; a raymarched hit has
// no vertex stage, so the march result fills them instead. Same names/types as
// SolidRenderer.frag's inputs so the shared shading body compiles unchanged.
vec2 fragUV;
vec3 fragNormal;
vec3 fragPosWorld;
ivec3 fragTexIndices;
ivec3 fragTextureTypes;
vec4 fragPosLightSpace;
vec3 fragPosWorldNotDisplaced;
vec3 fragTexWeights;
vec3 fragHSV;
vec3 fragSharpNormal;
vec3 fragTessLevel;

// Global toggles (same as SolidRenderer.frag).
bool roughnessEnabled = (ubo.roughnessEnabled != 0u);
bool aoEnabled = (ubo.ambientOcclusionEnabled != 0u);

// GPU SDF cell solver (QEF position + analytic normal) and the LandBrush /
// NormalBrush material resolver used by the octree topology pass.
#include "../../includes/octree/SdfCell.glsl"
#include "../../includes/octree/LandBrush.glsl"

// Solid surface shading (extracted): writes outColor; debug branches return.
#include "../../includes/solid/SolidSurface.glsl"

const int RAYMARCH_DISCARD_BRUSH = -1;
// Grazing rays through the depth-12 terrain leaf lattice can need ~800
// iteration steps (descent + empty/solid skips) before reaching the surface;
// 320 left visible holes wherever the budget ran out. The cost is only paid by
// pixels that actually take that many steps.
const int RM_MAX_STEPS = 1024;
const int RM_MAX_STACK = 40;
const int RM_BISECT_STEPS = 7;

float rmHitEpsilon(float nodeSize) {
    return max(nodeSize * 2.0e-3, 1.0e-4);
}

// Resolve a node's shading material at a given QEF position. Shared by the
// march's cache (position solved from the node) and the hit-time material
// blend (positions solved once for every ring cell).
int rmResolveMaterialAt(uint id, vec3 qefPos) {
    int raw = rmNodes[id].meta.z;
    if (raw != LAND_BRUSH_INDEX && raw != NORMAL_BRUSH) return raw;

    float sdf[8];
    for (int i = 0; i < 8; ++i) sdf[i] = rmNodeSdf(id, i);
    vec3 n = sdfCellNormal(sdf, rmNodeMin(id), rmNodeSize(id), qefPos);

    if (raw == LAND_BRUSH_INDEX) {
        // Mirror writeVertexOnce(): LAND_UNDERGROUND is a shading fallback,
        // not a topology discard.
        int land = landMaterialIndex(qefPos, n);
        return (land == LAND_UNDERGROUND) ? LAND_ROCK : land;
    }
    // NormalBrush keeps only +Y-facing cells, like the old CPU painter.
    return (dot(normalize(n), vec3(0.0, 1.0, 0.0)) > 0.0) ? 0 : RAYMARCH_DISCARD_BRUSH;
}

// Node QEF vertex position (the same value the raster vertex payload carries).
vec3 rmCellQef(uint id) {
    float sdf[8];
    for (int i = 0; i < 8; ++i) sdf[i] = rmNodeSdf(id, i);
    return sdfCellPosition(sdf, rmNodeMin(id), rmNodeSize(id));
}

// Final material of a node, bit-for-bit the OctreeTriangles.comp
// nodeResolvedBrush() + writeVertexOnce() rule. A LAND cell that classifies as
// underground is still meshed by the raster path and its vertex payload
// substitutes the steep-terrain material, so the raymarch does the same instead
// of discarding the hit. DISCARD_BRUSH (-1) from a NORMAL/unpainted cell means
// "no surface here".
int rmResolvedBrush(uint id) {
    int raw = rmNodes[id].meta.z;
    if (raw != LAND_BRUSH_INDEX && raw != NORMAL_BRUSH) return raw;
    return rmResolveMaterialAt(id, rmCellQef(id));
}

vec3 rmNodeNormal(uint id, vec3 p) {
    float sdf[8];
    for (int i = 0; i < 8; ++i) sdf[i] = rmNodeSdf(id, i);
    return sdfCellNormal(sdf, rmNodeMin(id), rmNodeSize(id), p);
}

// Per-invocation material cache. rmResolvedBrush() runs the QEF solver
// (sdfCellPosition + sdfCellNormal), so resolve it once per visited node instead
// of once per march step. GLSL globals here are invocation-local.
uint rmBrushNode = 0xFFFFFFFFu;
int rmBrushValue = RAYMARCH_DISCARD_BRUSH;
int rmCachedBrush(uint id) {
    if (id != rmBrushNode) {
        rmBrushNode = id;
        rmBrushValue = rmResolvedBrush(id);
    }
    return rmBrushValue;
}

// ── Tessellation displacement ("bump the SDF") ─────────────────────────────
// The raster path displaces every tessellated vertex along its smooth cell
// normal by (height - 0.5) * tessHeightScale. Fold the same signed offset into
// the marched field: f_bumped(p) = f_base(p) - disp(p). Since the base field is
// a signed distance, subtracting the offset moves the isosurface along the
// normal by exactly disp. The gates mirror the raster TES: displacement is only
// active when tessellation is enabled and the hit material has mapping on, so
// switching rasterization <-> raymarch (or tessellation on/off) keeps both
// surfaces aligned.
float rmDisplacementN(uint id, int material, vec3 p, vec3 n) {
    if (ubo.tessellationEnabled == 0u || material < 0) return 0.0;
    if (!materialNamed(materials[material]).mappingEnabled) return 0.0;
    int texType = int(rmNodes[id].hsv0.w + 0.5);
    vec2 uv = rmNodes[id].hsv1.xy;
    return materialDisplacementLod(material, texType, uv, p, n);
}

float rmDisplacement(uint id, int material, vec3 p) {
    return rmDisplacementN(id, material, p, rmNodeNormal(id, p));
}

// Bumped trilinear field. `base` receives the unbumped value (used for the
// conservative advance outside the displacement bound); `maxDisp` is the
// maximum |offset| this material can produce (0 = no bump).
float rmField(uint id, vec3 p, int material, out float base, out float maxDisp) {
    base = rmTrilinear(id, p);
    maxDisp = 0.0;
    if (ubo.tessellationEnabled == 0u || material < 0 || !rmIsFinite(base)) return base;
    MaterialNamed m = materialNamed(materials[material]);
    if (!m.mappingEnabled) return base;
    maxDisp = 0.5 * abs(m.tessHeightScale);
    return base - rmDisplacement(id, material, p);
}

// Tessellation level the raster TCS would request for this material at this
// world position (computeEdgeTess with the hit's distance). Feeds the
// DEBUG_MODE_TESS_HEAT view and sizes the bump's fine march step; the raymarch
// itself has no tessellated vertices.
float rmTessLevel(uint id, int material, vec3 p) {
    if (ubo.tessellationEnabled == 0u || material < 0) return 1.0 / 16.0;
    MaterialNamed m = materialNamed(materials[material]);
    if (!m.mappingEnabled) return 1.0 / 16.0;
    float d = length(p - ubo.viewPosition);
    float t = clamp(1.0 - smoothstep(ubo.tessNearDist, ubo.tessFarDist, d), 0.0, 1.0);
    float minLevel = m.minLevel * ubo.tessellationFactor;
    float maxLevel = m.maxLevel * ubo.tessellationFactor;
    return clamp((mix(minLevel, maxLevel, t) + m.tessLevel * t) / 16.0, 0.0, 4.0);
}

// ── Hit-time brush blending (raster TCS parity) ────────────────────────────
// The raster path compresses the triangle's up-to-three corner materials into
// unique slots (SolidRenderer.tesc) and blends them with the barycentric
// weights of the fragment inside the emitted triangle. A raymarched hit has no
// vertex stream, so reconstruct the same local Surface-Nets triangle:
//   * find the cell edge nearest the hit that carries a sign change,
//   * sample the four quadrant cells around it exactly like sampleQuadrants()
//     in OctreeTriangles.comp,
//   * use their QEF vertices — displaced by the same per-vertex tessellation
//     offset the raster TES applies — as the triangle corners and barycentric
//     weights, falling back to inverse-distance blending when the hit lands
//     slightly off the dual mesh.
// This makes material transitions across cell boundaries as smooth as the
// raster mesh instead of snapping to the hit cell's brush.

const uint RM_INVALID_NODE = 0xFFFFFFFFu;

struct RmSurfaceBlend {
    ivec3 indices; // up to three unique material indices (unused = index 0)
    ivec3 types;   // parallel per-vertex texture types
    vec3 weights;  // slot blend weights (sum = 1)
    vec2 uv;       // interpolated per-vertex UV scale
    vec3 hsv;      // interpolated per-vertex HSV tint
};

// Same root descent as OctreeTriangles.comp findCellAt, over the flattened
// raymarch table. Stops at the deepest stored surface node containing `pos`;
// absent (non-surface/pruned) children stop at the parent, matching the
// raymarch field's own LoD fallback. `nonSurfaceStop` reports that the descent
// ended because the point lies in a Solid/Empty/absent child, in which case the
// caller must not treat the parent as a surface ring vertex (findCellAt would
// have returned that non-surface cell and isSurfaceCell would have rejected it).
uint rmFindNodeAt(uint root, vec3 pos, out bool nonSurfaceStop) {
    nonSurfaceStop = false;
    vec3 rootMin = rmNodeMin(root);
    float rootSize = rmNodeSize(root);
    if (any(lessThan(pos, rootMin)) || any(greaterThan(pos, rootMin + vec3(rootSize)))) {
        return RM_INVALID_NODE;
    }
    uint idx = root;
    for (int i = 0; i < RM_MAX_STACK; ++i) {
        if (rmNodeIsLeaf(idx)) break;
        vec3 mn = rmNodeMin(idx);
        float halfSize = rmNodeSize(idx) * 0.5;
        int slot = (pos.x >= mn.x + halfSize ? 4 : 0)
                 + (pos.y >= mn.y + halfSize ? 2 : 0)
                 + (pos.z >= mn.z + halfSize ? 1 : 0);
        int child = rmNodeChild(idx, slot);
        if (child < 0) {
            // Absent slot: a pruned Surface child keeps the parent's field
            // (raymarch fallback); every other absent slot is a non-surface
            // region (Empty/Solid or a surface-push failure) and is not a ring
            // vertex, exactly like a non-surface cell in the raster walker.
            if (!rmNodeChildPruned(idx, slot)) nonSurfaceStop = true;
            break;
        }
        idx = uint(child);
    }
    return idx;
}

// EdgeCell::isSurface() + nodeTopologyAllowed() from OctreeTriangles.comp: a
// ring vertex must be a Surface node whose own corners cross zero and whose
// paint is not a raw discard. (Infinity corners are skipped by sdfEvalType.)
bool rmBlendCellValid(uint id) {
    if (rmNodes[id].meta.y != RM_TYPE_SURFACE) return false;
    float sdf[8];
    for (int i = 0; i < 8; ++i) sdf[i] = rmNodeSdf(id, i);
    if (sdfEvalType(sdf) != SDF_TYPE_SURFACE) return false;
    int raw = rmNodes[id].meta.z;
    if (raw == NORMAL_BRUSH) return rmResolvedBrush(id) > RAYMARCH_DISCARD_BRUSH;
    return raw > RAYMARCH_DISCARD_BRUSH;
}

// ownerLess(): cube size, min, then table identity (OctreeTriangles.comp uses
// the same rule so both sides of a boundary pick the same owner).
bool rmOwnerLess(uint a, uint b) {
    float sa = rmNodeSize(a);
    float sb = rmNodeSize(b);
    if (sa != sb) return sa < sb;
    vec3 ma = rmNodeMin(a);
    vec3 mb = rmNodeMin(b);
    if (ma.x != mb.x) return ma.x < mb.x;
    if (ma.y != mb.y) return ma.y < mb.y;
    if (ma.z != mb.z) return ma.z < mb.z;
    return a < b;
}

void rmEdgeAxes(int axis, out int u, out int v) {
    if (axis == 0) { u = 1; v = 2; }
    else if (axis == 1) { u = 0; v = 2; }
    else { u = 0; v = 1; }
}

// Quadrant sign pairs (SIGNS in OctreeTriangles.comp).
void rmQuadrantSigns(int axis, int quadrant, out int su, out int sv) {
    const int qs[24] = int[24](
        -1, -1,  -1, 1,   1, 1,   1, -1,
        -1, -1,   1, -1,  1, 1,  -1, 1,
        -1, -1,  -1, 1,   1, 1,   1, -1
    );
    su = qs[axis * 8 + quadrant * 2 + 0];
    sv = qs[axis * 8 + quadrant * 2 + 1];
}

// Barycentric coordinates of p on triangle (a,b,c), least-squares projected
// onto its plane (works when the hit is slightly off the dual mesh plane).
bool rmBary(vec3 p, vec3 a, vec3 b, vec3 c, out vec3 w) {
    vec3 v0 = b - a;
    vec3 v1 = c - a;
    vec3 v2 = p - a;
    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    float den = d00 * d11 - d01 * d01;
    if (abs(den) < 1e-12) {
        w = vec3(1.0, 0.0, 0.0);
        return false;
    }
    float v = (d11 * d20 - d01 * d21) / den;
    float u = (d00 * d21 - d01 * d20) / den;
    w = vec3(1.0 - v - u, v, u);
    return true;
}

// Default blend: the hit cell's own material/type/UV/HSV (previous behavior).
RmSurfaceBlend rmDefaultBlend(uint id) {
    RmSurfaceBlend b;
    int m = rmCachedBrush(id);
    b.indices = ivec3(max(m, 0));
    b.types = ivec3(int(rmNodes[id].hsv0.w + 0.5));
    b.weights = vec3(1.0, 0.0, 0.0);
    b.uv = rmNodes[id].hsv1.xy;
    b.hsv = rmNodes[id].hsv0.xyz;
    return b;
}

RmSurfaceBlend rmBlendAtHit(uint root, uint hitId, vec3 baseHit, vec3 displacedHit) {
    RmSurfaceBlend outBlend = rmDefaultBlend(hitId);

    float sz = rmNodeSize(hitId);
    vec3 mn = rmNodeMin(hitId);

    // Nearest crossing edge of the hit cell. The hit cell's field crosses zero,
    // so at least one of the 12 edges has a sign change. The dual polygon that
    // covers the hit is generated around the edge closest to it. Edge selection
    // uses the UNDISPLACED base point: the topology/material corners come from
    // the base mesh, while the barycentric weights below use the displaced
    // corners the raster actually draws.
    int bestAxis = -1;
    int bestU = 0, bestV = 0;
    float bestFixedU = 0.0, bestFixedV = 0.0;
    float bestStart = 0.0, bestEnd = 0.0;
    float bestPos = 0.0;
    float bestDist = 1e30;
    for (int e = 0; e < 12; ++e) {
        ivec2 ec = SDF_CELL_EDGES[e];
        float d0 = rmNodeSdf(hitId, ec.x);
        float d1 = rmNodeSdf(hitId, ec.y);
        if ((d0 < 0.0) == (d1 < 0.0)) continue;

        vec3 p0 = mn + SDF_CELL_CORNERS[ec.x] * sz;
        vec3 p1 = mn + SDF_CELL_CORNERS[ec.y] * sz;
        vec3 d = abs(p1 - p0);
        int axis = (d.x >= d.y && d.x >= d.z) ? 0 : ((d.y >= d.x && d.y >= d.z) ? 1 : 2);
        int u, v;
        rmEdgeAxes(axis, u, v);

        float fixedU = p0[u];
        float fixedV = p0[v];
        float start = min(p0[axis], p1[axis]);
        float end = max(p0[axis], p1[axis]);
        float t = clamp(baseHit[axis], start, end);
        float du = baseHit[u] - fixedU;
        float dv = baseHit[v] - fixedV;
        float dist = du * du + dv * dv;
        if (dist < bestDist) {
            bestDist = dist;
            bestAxis = axis;
            bestU = u;
            bestV = v;
            bestFixedU = fixedU;
            bestFixedV = fixedV;
            bestStart = start;
            bestEnd = end;
            bestPos = t;
        }
    }
    if (bestAxis < 0) return outBlend;

    // Same nudge as makeEdgeSpan(): small enough to stay inside a cell but
    // large enough to survive FTZ on the axis planes.
    float offset = max(sz * 1e-3, 1e-6);
    float tSample = clamp(bestPos, bestStart + offset, bestEnd - offset);
    if (bestStart + offset >= bestEnd - offset) {
        tSample = 0.5 * (bestStart + bestEnd);
    }

    // sampleQuadrants(): four cells around the edge, deduped by node/position.
    // Each corner keeps the raster vertex payload (QEF position, material,
    // type, UV scale, HSV) and its per-vertex tessellation-displaced position,
    // which is what the raster triangle's corners actually are.
    uint ringId[4];
    vec3 ringBase[4];
    vec3 ringPos[4];
    int ringMat[4];
    int ringType[4];
    vec2 ringUv[4];
    vec3 ringHsv[4];
    int ringCount = 0;
    float dedupEps = max(sz * 1e-4, 1e-6);
    for (int q = 0; q < 4; ++q) {
        int su, sv;
        rmQuadrantSigns(bestAxis, q, su, sv);
        vec3 sp = vec3(0.0);
        sp[bestAxis] = tSample;
        sp[bestU] = bestFixedU + (su < 0 ? -offset : offset);
        sp[bestV] = bestFixedV + (sv < 0 ? -offset : offset);

        bool nonSurfaceStop;
        uint id = rmFindNodeAt(root, sp, nonSurfaceStop);
        if (nonSurfaceStop || id == RM_INVALID_NODE || !rmBlendCellValid(id)) continue;
        vec3 qp = rmCellQef(id);
        bool duplicate = false;
        for (int k = 0; k < ringCount; ++k) {
            vec3 dq = qp - ringBase[k];
            if (ringId[k] == id || dot(dq, dq) <= dedupEps * dedupEps) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate && ringCount < 4) {
            int m = rmResolveMaterialAt(id, qp);
            vec3 n = rmNodeNormal(id, qp);
            float disp = rmDisplacementN(id, m, qp, n);
            ringId[ringCount] = id;
            ringBase[ringCount] = qp;
            ringPos[ringCount] = qp + n * disp;
            ringMat[ringCount] = m;
            ringType[ringCount] = int(rmNodes[id].hsv0.w + 0.5);
            ringUv[ringCount] = rmNodes[id].hsv1.xy;
            ringHsv[ringCount] = rmNodes[id].hsv0.xyz;
            ++ringCount;
        }
    }
    if (ringCount < 3) return outBlend;

    // Rotate the polygon so the shared owner leads (identical rule to
    // emitSegment()): both sides of a chunk boundary then pick the same
    // triangle fan and material slot order.
    int owner = 0;
    for (int i = 1; i < ringCount; ++i) {
        if (rmOwnerLess(ringId[i], ringId[owner])) owner = i;
    }
    if (owner > 0) {
        uint rid[4]; vec3 rbase[4]; vec3 rpos[4];
        int rmat[4]; int rtype[4]; vec2 ruv[4]; vec3 rhsv[4];
        for (int i = 0; i < ringCount; ++i) {
            int k = (i + owner) % ringCount;
            rid[i] = ringId[k];
            rbase[i] = ringBase[k];
            rpos[i] = ringPos[k];
            rmat[i] = ringMat[k];
            rtype[i] = ringType[k];
            ruv[i] = ringUv[k];
            rhsv[i] = ringHsv[k];
        }
        for (int i = 0; i < ringCount; ++i) {
            ringId[i] = rid[i];
            ringBase[i] = rbase[i];
            ringPos[i] = rpos[i];
            ringMat[i] = rmat[i];
            ringType[i] = rtype[i];
            ringUv[i] = ruv[i];
            ringHsv[i] = rhsv[i];
        }
    }

    // Barycentric blend on the same fan the raster emits: (0,1,2) and, for a
    // four-quadrant ring, (0,2,3). Choose the triangle containing the hit (the
    // least-negative minimum weight is the fallback when neither strictly does).
    int ci0 = 0, ci1 = 1, ci2 = 2;
    vec3 triW = vec3(1.0, 0.0, 0.0);
    bool haveWeights = false;
    if (ringCount == 3) {
        haveWeights = rmBary(displacedHit, ringPos[0], ringPos[1], ringPos[2], triW);
    } else {
        vec3 w0, w1;
        bool ok0 = rmBary(displacedHit, ringPos[0], ringPos[1], ringPos[2], w0);
        bool ok1 = rmBary(displacedHit, ringPos[0], ringPos[2], ringPos[3], w1);
        float min0 = ok0 ? min(min(w0.x, w0.y), w0.z) : -1e9;
        float min1 = ok1 ? min(min(w1.x, w1.y), w1.z) : -1e9;
        if (ok0 && (min0 >= -0.02 || min1 < min0)) {
            ci0 = 0; ci1 = 1; ci2 = 2;
            triW = w0;
            haveWeights = true;
        } else if (ok1) {
            ci0 = 0; ci1 = 2; ci2 = 3;
            triW = w1;
            haveWeights = true;
        }
    }
    if (haveWeights && min(min(triW.x, triW.y), triW.z) < -0.5) {
        haveWeights = false; // wildly off the fan: use IDW below
    }

    if (!haveWeights) {
        // Inverse-distance fallback over the ring cells (still blends the
        // local brushes; only the weight shape differs from barycentrics).
        float dw[4];
        float eps2 = dedupEps * dedupEps;
        for (int i = 0; i < ringCount; ++i) {
            vec3 dq = ringPos[i] - displacedHit;
            dw[i] = 1.0 / (dot(dq, dq) + eps2);
        }
        // Keep the three strongest quadrants (the raster never blends four).
        int i0 = 0, i1 = 1, i2 = 2;
        if (ringCount == 4) {
            int worst = 0;
            if (dw[1] < dw[worst]) worst = 1;
            if (dw[2] < dw[worst]) worst = 2;
            if (dw[3] > dw[worst]) {
                if (worst == 0) i0 = 3;
                else if (worst == 1) i1 = 3;
                else i2 = 3;
            }
        }
        ci0 = i0; ci1 = i1; ci2 = i2;
        triW = vec3(dw[i0], dw[i1], dw[i2]);
        float wsum = triW.x + triW.y + triW.z;
        if (wsum <= 1e-12) return outBlend;
        triW /= wsum;
    }

    // Clamp/renormalize the barycentrics (a hit can sit marginally outside the
    // dual triangle because the march samples the trilinear field).
    vec3 w = max(triW, vec3(0.0));
    float wsum = w.x + w.y + w.z;
    if (wsum <= 1e-12) return outBlend;
    w /= wsum;

    int mats[3] = int[3](ringMat[ci0], ringMat[ci1], ringMat[ci2]);
    int types[3] = int[3](ringType[ci0], ringType[ci1], ringType[ci2]);
    vec2 uvs[3] = vec2[3](ringUv[ci0], ringUv[ci1], ringUv[ci2]);
    vec3 hsvs[3] = vec3[3](ringHsv[ci0], ringHsv[ci1], ringHsv[ci2]);

    // Unique-slot compression, mirroring SolidRenderer.tesc exactly: slot 0 is
    // corner 0, slot 1 the first differing corner 1, slot 2 the first differing
    // corner 2. Corners with negative material indices get zero weight.
    int u0 = (mats[0] >= 0) ? mats[0] : -1;
    int u1 = -1;
    int u2 = -1;
    if (mats[1] >= 0 && mats[1] != u0) u1 = mats[1];
    if (mats[2] >= 0 && mats[2] != u0 && mats[2] != u1) u2 = mats[2];

    vec3 slotW = vec3(0.0);
    for (int i = 0; i < 3; ++i) {
        if (mats[i] < 0) continue;
        if (mats[i] == u0) slotW.x += w[i];
        else if (mats[i] == u1) slotW.y += w[i];
        else slotW.z += w[i];
    }
    float slotSum = slotW.x + slotW.y + slotW.z;
    if (slotSum <= 1e-12 || u0 < 0) return outBlend;
    slotW /= slotSum;

    int type0 = (mats[0] >= 0) ? types[0] : 0;
    int type1 = (mats[1] >= 0 && mats[1] != u0) ? types[1] : 0;
    int type2 = (mats[2] >= 0 && mats[2] != u0 && mats[2] != u1) ? types[2] : 0;

    outBlend.indices = ivec3(u0, (u1 >= 0) ? u1 : u0, (u2 >= 0) ? u2 : u0);
    outBlend.types = ivec3(type0, (u1 >= 0) ? type1 : type0,
                           (u2 >= 0) ? type2 : type0);
    outBlend.weights = slotW;
    outBlend.uv = uvs[0] * w.x + uvs[1] * w.y + uvs[2] * w.z;
    outBlend.hsv = hsvs[0] * w.x + hsvs[1] * w.y + hsvs[2] * w.z;
    return outBlend;
}

// ── Octree SDF debug view helpers (DEBUG_MODE_OCTREE_SDF_RAYMARCH) ─────────
// Stable per-layer hue: Layer::Id is a stable handle, so a layer keeps its
// colour across table rebuilds. Golden-ratio hue spacing keeps neighbouring
// ids far apart.
vec3 rmLayerDebugColor(int layerId) {
    float hue = fract(float(layerId) * 0.618033988749895) * 360.0;
    return hsvToRgb(vec3(hue, 0.68, 0.95));
}

// Cheap integer hash in [0,1) so adjacent leaf cells get different tints.
float rmHash01(uint x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return float(x & 0x00ffffffu) * (1.0 / 16777216.0);
}

// Cell-local coordinates (0..1 per axis) of the leaf cube that produced the
// hit. rmNodeMin/rmNodeSize are the exact cube the CPU stored for this node.
vec3 rmCellLocal(uint node, vec3 p) {
    return clamp((p - rmNodeMin(node)) / max(rmNodeSize(node), 1e-8), 0.0, 1.0);
}

// Distance to the nearest of the 12 leaf-cube edges in cell-local space. Using
// edges (not the 6 faces) is deliberate: a surface that lies exactly on a cell
// face would otherwise cover the whole face with a "grid line". An edge is
// close only where the two nearest face distances are both small, so the
// lattice stays a thin outline along the surface patch's cell rim.
float rmCellEdge(vec3 l) {
    vec3 f = min(l, 1.0 - l); // distances to the six faces
    float a = min(f.x, min(f.y, f.z));
    float c = max(f.x, max(f.y, f.z));
    float b = f.x + f.y + f.z - a - c;
    float edgeDist = sqrt(a * a + b * b);
    return 1.0 - smoothstep(0.0, 0.06, edgeDist);
}

// Refinement depth: 0 at the layer root, 1 for cells 2^16 smaller. Deeper
// leaves are brighter so the refinement gradient is visible.
float rmCellDepth(uint rootNode, uint node) {
    float rootSize = rmNodeSize(rootNode);
    float nodeSize = rmNodeSize(node);
    if (rootSize <= 0.0 || nodeSize <= 0.0) return 0.0;
    return clamp(log2(rootSize / max(nodeSize, 1e-8)) / 16.0, 0.0, 1.0);
}

// March one layer root. Adaptive descent: the ray is confined to the current
// cell's [t, nodeExit] interval and follows stored Surface children. Empty and
// Solid children are skipped (neither can contain an isosurface crossing);
// only node-cap-pruned Surface children fall back to the parent trilinear
// field. Within a cell the field is sphere-traced with a 0.5 safety factor and
// a sign-change bisection refines the crossing. Discarded (underground / -Y
// NormalBrush) surfaces are skipped and the march continues.
bool rmMarchRoot(uint root, vec3 ro, vec3 rd,
                 out uint hitNode, out float hitT, out int hitBrush) {
    float t0, t1;
    if (!rmRayBox(rmNodeMin(root), rmNodeSize(root), ro, rd, t0, t1)) return false;
    float t = max(t0, 0.0);
    if (t >= t1) return false;

    uint stackNode[RM_MAX_STACK];
    float stackExit[RM_MAX_STACK];
    int depth = 0;

    uint node = root;
    float nodeExit = t1;

    for (int iter = 0; iter < RM_MAX_STEPS; ++iter) {
        // Left the current cell: resume the parent (or stop at the root).
        if (t >= nodeExit) {
            if (depth == 0) return false;
            --depth;
            node = stackNode[depth];
            nodeExit = stackExit[depth];
            continue;
        }

        vec3 p = ro + rd * t;
        float nodeSize = rmNodeSize(node);

        // The step below must never jump over the child cell that contains p:
        // stored Surface children are only noticed when the sample lands inside
        // them, and a thin front-face shell is much smaller than the distance
        // step that a coarse parent field (pruned-children fallback) produces.
        // Clamp every step to this child's exit so the ray visits the next cell
        // (and descends into a Surface child) instead of skipping it.
        float cellExit = nodeExit;

        // Descend into the child cell containing p. Select the cell at a small
        // forward nudge: after a cell skip/exit t lands exactly on a shared
        // face, where floor(local*2) rounds back into the cell being left and
        // the child ray-box test can report c1<=0. The default cellExit then
        // stays at the parent's exit, so the same node is re-entered with a
        // far-too-large interval and every sibling in between is skipped (the
        // observed cause of the missing terrain). The nudge is negligible at
        // cell scale but strictly inside the cell the ray is about to enter.
        if (!rmNodeIsLeaf(node)) {
            vec3 mn = rmNodeMin(node);
            float nudge = max(nodeSize * 1.0e-5, 1.0e-4);
            float tn = t + nudge;
            vec3 pn = ro + rd * tn;
            vec3 local = (pn - mn) / max(nodeSize, 1e-8);
            ivec3 cell = clamp(ivec3(floor(local * 2.0)), ivec3(0), ivec3(1));
            int slot = cell.x * 4 + cell.y * 2 + cell.z;
            int child = rmNodeChild(node, slot);

            float childSize = nodeSize * 0.5;
            vec3 childMin = mn + vec3(cell) * childSize;

            // Exit of the child cell measured from the nudged point. It is
            // strictly inside the cell, so the slab test is well defined even
            // when the un-nudged ray sits on a face.
            float c0, c1;
            if (rmRayBox(childMin, childSize, pn, rd, c0, c1) && c1 > 0.0) {
                cellExit = min(nodeExit, tn + c1);
            } else {
                // Degenerate slab result: advance a hair and re-evaluate the
                // cell next iteration instead of skipping to the parent exit.
                t = min(t + nudge, nodeExit);
                continue;
            }

            if (child >= 0) {
                if (depth < RM_MAX_STACK) {
                    stackNode[depth] = node;
                    stackExit[depth] = nodeExit;
                    ++depth;
                    node = uint(child);
                    nodeExit = cellExit;
                    continue; // p is already inside the child: keep t
                }
            } else if (!rmNodeChildPruned(node, slot) &&
                       rmNodeChildType(node, slot) != RM_TYPE_SURFACE) {
                // Empty (all-outside) or Solid (all-inside): the unbumped
                // isosurface cannot pass through, so skip to the child's
                // ray-box exit without sampling. Pruned SURFACE children also
                // have no stored index, but their pruned bit keeps them out of
                // this branch: they fall through to the parent field below.
                //
                // Tessellation bump: base - disp CAN cross inside this child
                // even though the child's own corners are all one sign (the
                // bump pushes the rasterized surface across cell borders), so
                // while this node's material is displaced, keep sampling the
                // parent field instead of skipping. Without an active bump the
                // skip is exact and stays enabled.
                int bumpBrush = rmCachedBrush(node);
                bool bumpActive = ubo.tessellationEnabled != 0u
                    && bumpBrush >= 0
                    && materialNamed(materials[bumpBrush]).mappingEnabled
                    && materialNamed(materials[bumpBrush]).tessHeightScale != 0.0;
                if (!bumpActive) {
                    t = max(cellExit, t + nudge); // nudge guards a zero-length skip
                    continue;
                }
            }
            // Pruned surface child (also: stack exhausted): keep sampling the
            // parent field, but only up to the end of this child cell.
        }

        int material = rmCachedBrush(node);
        float baseD;
        float maxDisp;
        float d = rmField(node, p, material, baseD, maxDisp);
        if (isnan(d)) d = 1.0; // defensive: never let a bad sample freeze the march
        float eps = rmHitEpsilon(nodeSize);

        if (abs(d) <= eps) {
            if (material != RAYMARCH_DISCARD_BRUSH) {
                hitNode = node;
                hitT = t;
                hitBrush = material;
                return true;
            }
            t += eps;
            continue;
        }

        // Sphere trace. Without displacement the trilinear field is (locally)
        // a signed distance and abs(d) * 0.5 is safe. With the tessellation
        // bump the field is base - disp: outside |base| <= maxDisp it cannot
        // have crossed yet, so advance by the guaranteed-positive remainder;
        // inside that band use the material's tessellation spacing (the
        // density the rasterizer actually resolves) so bump features are not
        // skipped. The child-cell exit still bounds every step.
        float stepLen;
        if (maxDisp > 0.0) {
            float safe = abs(baseD) - maxDisp;
            if (safe > eps) {
                stepLen = max(safe * 0.5, eps);
            } else {
                // Match the raster's actual subdivision at this distance:
                // rmTessLevel is the TCS level / 16 it would write, so
                // level * 16 is the vertex spacing denominator.
                float levels = clamp(rmTessLevel(node, material, p) * 16.0, 1.0, 64.0);
                float fineStep = max(nodeSize / levels, 1e-3);
                stepLen = max(min(abs(d) * 0.5, fineStep), eps);            }
        } else {
            stepLen = max(abs(d) * 0.5, eps);
        }
        float tNext = min(t + stepLen, cellExit);
        float dNextBase;
        float dNextMax;
        float dNext = rmField(node, ro + rd * tNext, material, dNextBase, dNextMax);
        if (isnan(dNext)) dNext = -1.0;

        // Sign change inside this cell: refine with bisection.
        if (d > 0.0 && dNext <= 0.0) {
            float lo = t;
            float hi = tNext;
            for (int b = 0; b < RM_BISECT_STEPS; ++b) {
                float mid = 0.5 * (lo + hi);
                float bBase;
                float bMax;
                float dm = rmField(node, ro + rd * mid, material, bBase, bMax);
                if (isnan(dm)) break;
                if (dm > 0.0) lo = mid; else hi = mid;
            }
            float tHit = 0.5 * (lo + hi);
            if (material != RAYMARCH_DISCARD_BRUSH) {
                hitNode = node;
                hitT = tHit;
                hitBrush = material;
                return true;
            }
            t = tHit + eps;
            continue;
        }

        t = tNext;
    }
    return false;
}

void main() {
    vec2 extent = max(pc.viewport.xy, vec2(1.0));
    vec2 ndc = gl_FragCoord.xy / extent * 2.0 - 1.0;

    vec4 farPoint = ubo.invViewProjection * vec4(ndc, 1.0, 1.0);
    if (abs(farPoint.w) < 1e-8) discard;

    vec3 ro = ubo.viewPosition;
    vec3 rd = normalize(farPoint.xyz / farPoint.w - ro);

    // Nearest hit across every visible Solid layer root.
    float nearestT = 1.0e30;
    uint nearestNode = 0u;
    uint nearestRoot = 0u;
    int nearestBrush = RAYMARCH_DISCARD_BRUSH;
    bool found = false;
    for (uint i = 0u; i < pc.rootCount; ++i) {
        uint node;
        float t;
        int brush;
        if (rmMarchRoot(rmRoots[i].node, ro, rd, node, t, brush) && t < nearestT) {
            nearestT = t;
            nearestNode = node;
            nearestRoot = i;
            nearestBrush = brush;
            found = true;
        }
    }
    if (!found || nearestBrush == RAYMARCH_DISCARD_BRUSH) discard;

    vec3 hitPos = ro + rd * nearestT;

    // Material / mapping payload of the hit cell. Until the local ring is
    // reconstructed this is the hit cell's own brush; rmBlendAtHit() replaces
    // it with the same up-to-three-material barycentric blend the raster TCS
    // produces for the triangle under the hit.
    int material = max(nearestBrush, 0);

    // Tessellation parity: the raster TES samples the height maps at the
    // UNDISPLACED vertex and moves it along the smooth cell normal. Recover
    // that base point (and its normal) from the bumped hit so shading and
    // mapping are evaluated at the same world position the raster path uses.
    // Two fixed-point steps; the offset is small relative to the height-map
    // feature scale, so this converges to the raster's undisplaced vertex.
    vec3 sdfNormal = rmNodeNormal(nearestNode, hitPos);
    float hitDisplacement = rmDisplacement(nearestNode, material, hitPos);
    vec3 basePos = hitPos - sdfNormal * hitDisplacement;
    for (int i = 0; i < 2 && hitDisplacement != 0.0; ++i) {
        sdfNormal = rmNodeNormal(nearestNode, basePos);
        hitDisplacement = rmDisplacement(nearestNode, material, basePos);
        basePos = hitPos - sdfNormal * hitDisplacement;
    }
    vec3 normal = sdfNormal;
    if (dot(normal, rd) > 0.0) normal = -normal; // front surface only

    // The raster material blend belongs to the base (undisplaced) cell: use
    // that cell when it is in the table, otherwise the bumped hit's own cell.
    uint rootNode = rmRoots[nearestRoot].node;
    bool baseCellNonSurface;
    uint blendCell = rmFindNodeAt(rootNode, basePos, baseCellNonSurface);
    if (blendCell == RM_INVALID_NODE || baseCellNonSurface) blendCell = nearestNode;
    RmSurfaceBlend blend = rmBlendAtHit(rootNode, blendCell, basePos, hitPos);

    fragPosWorld = hitPos;
    fragPosWorldNotDisplaced = basePos;
    fragNormal = normal;
    fragSharpNormal = normal;
    fragTexIndices = blend.indices;
    fragTextureTypes = blend.types;
    fragTexWeights = blend.weights;
    fragHSV = blend.hsv;
    // Interpolated per-vertex UV scale (OctreeNode::data), matching the
    // raster's interpolated fragUV.
    fragUV = blend.uv;
    fragPosLightSpace = ubo.lightSpaceMatrix * vec4(hitPos, 1.0);
    // Same feed the TES writes (tess levels / 16, clamped to the heat range).
    // The raster's tess levels are per base cell, not per bumped hit.
    fragTessLevel = vec3(rmTessLevel(blendCell, rmCachedBrush(blendCell), basePos));

    vec4 clip = ubo.viewProjection * vec4(hitPos, 1.0);
    gl_FragDepth = (clip.w > 1e-6) ? clamp(clip.z / clip.w, 0.0, 1.0) : 1.0;

    // Per-layer SDF inspection: tint the raymarched isosurface with the owning
    // layer's hue and overlay the leaf cell that produced the hit. This view
    // deliberately bypasses shading so the octree discretization (cell size,
    // per-leaf identity, pruned-node fallback, missing corners) is what the
    // image shows. Only the solid raymarch path can produce it (the mode
    // forces that path in MyApp).
    if (ubo.debugMode == DEBUG_MODE_OCTREE_SDF_RAYMARCH) {
        vec3 layerColor = rmLayerDebugColor(rmRoots[nearestRoot].layerId);
        float depth = rmCellDepth(rmRoots[nearestRoot].node, nearestNode);
        vec3 color = layerColor * (0.60 + 0.40 * depth);

        // Per-leaf identity so neighbouring cells are distinguishable even on
        // locally flat ground.
        color *= rmNodeIsLeaf(nearestNode) ? (0.70 + 0.30 * rmHash01(nearestNode)) : 0.55;

        // Leaf-cell outline: the edge of the cell's surface patch. The outer
        // lines are the shared cell edges, so the lattice exposes the cell size
        // without turning whole axis-aligned faces white.
        vec3 cell = rmCellLocal(nearestNode, hitPos);
        color = mix(color, vec3(1.0), 0.65 * rmCellEdge(cell));

        // Non-leaf hits only occur when a surface child was pruned by NODE_CAP
        // (or the stack budget ran out): the march falls back to this node's
        // trilinear field. Flag them magenta with a diagonal hatch so coarsened
        // regions cannot be mistaken for real leaf detail.
        if (!rmNodeIsLeaf(nearestNode)) {
            float hatch = step(0.5, fract((cell.x + cell.y + cell.z) * 8.0));
            color = mix(color, vec3(1.0, 0.05, 0.85), 0.55 + 0.20 * hatch);
        }

        // Cells with INFINITY corner sentinels interpolate over their finite
        // corners only; tint them blue by how much data is missing.
        int finiteCorners = 0;
        for (int c = 0; c < 8; ++c)
            finiteCorners += rmIsFinite(rmNodeSdf(nearestNode, c)) ? 1 : 0;
        if (finiteCorners < 8) {
            float missing = float(8 - finiteCorners) / 8.0;
            color = mix(color, vec3(0.10, 0.40, 1.00), 0.55 * missing);
        }

        outColor = vec4(clamp(color, 0.0, 1.0), 1.0);
        return;
    }

    outColor = vec4(0.0);
    // Zero override: shadeSolidSurface derives the facet normal from the
    // marched hit positions, exactly like the raster path derives it from the
    // displaced triangle positions (triplanar sign/weights match).
    shadeSolidSurface(vec3(0.0));
}

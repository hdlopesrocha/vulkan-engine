#version 460
// RT_ENABLED is defined for the OctreeRaymarchRT.frag.spv variant (hardware
// RT path). Without it the shader uses the procedural-sky fallback with
// identical miss baselines, so non-RT hardware stays validation-clean.
#ifdef RT_ENABLED
#extension GL_EXT_ray_query : require
#endif

// ── Octree SDF raymarcher ───────────────────────────────────────────────────
// Optional SDF path (Settings: "Octree SDF Raymarch", selected in the Ray
// Marching widget): instead of drawing the tessellated chunks of each layer,
// this fullscreen pass marches the octree's stored corner SDFs for EVERY
// terrain layer (Solid and Water) and shades the nearest isosurface hit. Solid
// layers go through the same shadeSolidSurface() routine the raster path uses
// (materials/mapping, CSM shadows, RT reflections/contact shadows); Water
// layers use a water-surface shading branch (Snell-refracted lake bed + SDF
// mirror ray, composed with the raster surface's Fresnel/strength mix).
//
// The node table (utils/OctreeRaymarchTable.hpp) stores SURFACE cells only,
// flattened breadth-first per enabled terrain layer. Each layer contributes one
// root entry with the root's conservative surface AABB; roots are sorted by
// AABB entry, marched front to back and each later march is clamped to the
// nearest hit so far. The nearest hit wins, so solid and water surfaces occlude
// each other correctly in one pass. Non-surface children are not stored: their
// SpaceType is packed in the parent so Empty children (all corners outside) and
// Solid children (all corners inside) are both skipped — by construction the
// unbumped isosurface cannot cross either. Only node-cap-pruned Surface
// children fall back to the parent's trilinear field. With tessellation
// displacement active the skip is suspended: the bump (below) can move the
// surface across a child border, so those children are sampled through the
// parent field to keep the marched surface continuous with the displaced
// raster triangles.
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
#include "../../includes/water/Gerstner.glsl"
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
    vec4 viewport;     // xy = solid pass target extent in pixels
    uint rootCount;    // number of layer roots in the set-2 root buffer
    uint tableVersion; // low 32 bits of the table revision (future caches)
    uint nodeCount;    // records uploaded for this frame slot
    uint flags;        // bit0: profile counters enabled
} pc;

#ifdef OCTREE_PROFILE
void rmCount(int idx) {
    if ((pc.flags & 1u) != 0u) atomicAdd(rmCounters[idx], 1u);
}
#else
void rmCount(int idx) { }
#endif

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
// Material + Lipschitz helpers shared with the once-per-revision payload pass
// (OctreeDerive.comp): the cached payload and this fallback use one rule.
#include "../../includes/octree/RaymarchMaterial.glsl"

// Solid surface shading (extracted): writes outColor; debug branches return.
#include "../../includes/solid/SolidSurface.glsl"

const int RAYMARCH_DISCARD_BRUSH = RM_MATERIAL_DISCARD;
// Grazing rays through the depth-12 terrain leaf lattice can need ~800
// iteration steps (descent + empty/solid skips) before reaching the surface;
// 320 left visible holes wherever the budget ran out. The cost is only paid by
// pixels that actually take that many steps.
const int RM_MAX_STEPS = 1024;
// The flattened table's measured maximum stored depth is 12 (the node cap
// truncates deeper levels); 16 leaves margin without paying for a 40-entry
// per-invocation scratch stack.
const int RM_MAX_STACK = 16;
// Root-interval sort budget. The layer list is two roots in practice; 16
// covers every realistic scene without turning the per-invocation interval
// array into a large scratch allocation. Roots past this budget keep the
// unsorted fallback loop.
const int RM_MAX_ROOTS = 16;
const int RM_BISECT_STEPS = 7;

// Distance-adaptive hit tolerance. The node epsilon is the base floor; with a
// screen-space tolerance enabled the band widens with the world size of one
// pixel at the sample distance, so distant and grazing rays stop as soon as
// the crossing is sub-pixel instead of over-resolving it. One pixel's world
// size at distance t is 2*tan(fovy/2)/height * t; viewProjection[1][1] holds
// 1/tan(fovy/2) (pc.viewport is the solid target extent in pixels).
float rmHitEpsilon(float nodeSize, float t) {
    float eps = max(nodeSize * 2.0e-3, 1.0e-4);
    float px = ubo.raymarchScreenEps;
    if (px <= 0.0) return eps;
    float pixelAngle = 2.0 / max(abs(ubo.viewProjection[1][1]) * max(pc.viewport.y, 1.0), 1e-6);
    return max(eps, px * max(t, 0.0) * pixelAngle);
}

// Resolve a node's shading material at a given QEF position. Shared by the
// march's cache (position solved from the node) and the hit-time material
// blend (positions solved once for every ring cell). The classifier itself
// lives in RaymarchMaterial.glsl so OctreeDerive.comp resolves the identical
// index into meta.x.
int rmResolveMaterialAt(uint id, vec3 qefPos) {
    int raw = rmNodes[id].meta.z;
    if (raw != LAND_BRUSH_INDEX && raw != NORMAL_BRUSH) return raw;

    rmCount(RM_COUNTER_MATERIAL_CLASSIFY);
    float sdf[8];
    for (int i = 0; i < 8; ++i) sdf[i] = rmNodeSdf(id, i);
    vec3 mn = rmNodeMin(id);
    float sz = rmNodeSize(id);
    vec3 n = sdfCellNormal(sdf, mn, sz, qefPos);
    return rmResolveMaterialAtCorners(raw, sdf, mn, sz, qefPos, n);
}

bool rmQefCached(uint id) {
    return (uint(rmNodes[id].meta.w) & RM_FLAG_QEF) != 0u;
}

// Node QEF vertex position (the same value the raster vertex payload carries).
// OctreeDerive.comp computed it once per revision; the fragment only solves it
// when the payload is unavailable (derive module missing / allocation failure).
vec3 rmCellQef(uint id) {
    if (rmQefCached(id)) {
        rmCount(RM_COUNTER_QEF_PAYLOAD);
        return rmPayload[id].qef.xyz;
    }
    rmCount(RM_COUNTER_QEF_SOLVES);
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
    // Payload cache: OctreeDerive.comp already resolved the material into
    // meta.x once per revision. Only slots whose derive pass never ran (module
    // missing) or nodes without the flag take the QEF fallback below.
    if ((uint(rmNodes[id].meta.w) & RM_FLAG_PAYLOAD) != 0u) return rmNodes[id].meta.x;
    int raw = rmNodes[id].meta.z;
    if (raw != LAND_BRUSH_INDEX && raw != NORMAL_BRUSH) return raw;
    return rmResolveMaterialAt(id, rmCellQef(id));
}

// Ring material lookup. The payload pass resolved the identical LandBrush /
// NormalBrush classifier result into meta.x, so the hit ring must not rerun the
// branchy Perlin classifier per quadrant cell. Cells without a payload (derive
// unavailable) keep the fragment-side classifier, using the ring's already
// solved QEF position instead of solving it again.
int rmRingMaterial(uint id, vec3 qp) {
    if ((uint(rmNodes[id].meta.w) & RM_FLAG_PAYLOAD) != 0u) return rmNodes[id].meta.x;
    return rmResolveMaterialAt(id, qp);
}

vec3 rmNodeNormal(uint id, vec3 p) {
    rmCount(RM_COUNTER_GRADIENT_EVALS);
    float sdf[8];
    for (int i = 0; i < 8; ++i) sdf[i] = rmNodeSdf(id, i);
    return sdfCellNormal(sdf, rmNodeMin(id), rmNodeSize(id), p);
}

// Normal at the node's QEF vertex. Exact consumers (the ring) can use the
// stored vector; hit-point consumers keep rmNodeNormal() because their sample
// position differs from the QEF vertex.
vec3 rmNodeNormalAtQef(uint id, vec3 qp) {
    if (rmQefCached(id)) return rmPayload[id].normal.xyz;
    return rmNodeNormal(id, qp);
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

bool rmDisplacementActive(int material) {
    if (ubo.tessellationEnabled == 0u || material < 0) return false;
    return materialNamed(materials[material]).mappingEnabled;
}

// Compute the normal only when a displacement can actually be produced. The
// old unconditional rmNodeNormal() here was a full gradient per hit (and a
// second one when the caller also needed the base-point normal).
float rmDisplacement(uint id, int material, vec3 p) {
    if (!rmDisplacementActive(material)) return 0.0;
    return rmDisplacementN(id, material, p, rmNodeNormal(id, p));
}

// Same, with a normal the caller already computed at p.
float rmDisplacement(uint id, int material, vec3 p, vec3 n) {
    if (!rmDisplacementActive(material)) return 0.0;
    return rmDisplacementN(id, material, p, n);
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

// Secant refinement of a hit accepted inside the (possibly screen-space
// widened) tolerance band. The march only knows the last two field samples of
// the same node, prevT/dPrev and t/d, which bracket the crossing when the band
// is wider than the step; a short secant iteration recovers the sub-band
// crossing, clamped to the band so the hit can never leave it. Only called
// when the screen-space tolerance is active; the plain node epsilon keeps the
// historical acceptance at the sampled t.
float rmRefineHit(uint node, vec3 ro, vec3 rd, int material,
                  float tPrev, float dPrev, float t, float d, float eps) {
    if (!(tPrev < t) || !(abs(t - tPrev) > 1e-8) || abs(dPrev) <= eps) return t;
    float t0 = tPrev, f0 = dPrev;
    float t1 = t, f1 = d;
    for (int i = 0; i < 2; ++i) {
        float den = f1 - f0;
        if (abs(den) < 1e-12) break;
        float tn = clamp(t1 - f1 * (t1 - t0) / den, min(t0, t1), max(t0, t1));
        tn = clamp(tn, t - eps, t + eps);   // never move off the hit band
        rmCount(RM_COUNTER_BISECT_EVALS);
        float bBase, bMax;
        float fn = rmField(node, ro + rd * tn, material, bBase, bMax);
        if (isnan(fn)) break;
        t0 = t1; f0 = f1; t1 = tn; f1 = fn;
        if (abs(f1) < 1e-6) break;
    }
    return clamp(t1, t - eps, t + eps);
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

// Phase 1: the nearest accepted hit's ancestor chain. rmMarchRoot copies its
// local stack here on acceptance, so the hit path can start its node queries
// from a containing ancestor instead of re-descending from the layer root.
// Invocation-local, like the material cache below.
uint rmHitStackNode[RM_MAX_STACK];
int rmHitStackDepth = 0;

struct RmSurfaceBlend {
    ivec3 indices; // up to three unique material indices (unused = index 0)
    ivec3 types;   // parallel per-vertex texture types
    vec3 weights;  // slot blend weights (sum = 1)
    vec2 uv;       // interpolated per-vertex UV scale
    vec3 hsv;      // interpolated per-vertex HSV tint
    vec3 normal;   // interpolated per-vertex base normal (zero = use march normal)
};

// Same root descent as OctreeTriangles.comp findCellAt, over the flattened
// raymarch table. Stops at the deepest stored surface node containing `pos`;
// absent (non-surface/pruned) children stop at the parent, matching the
// raymarch field's own LoD fallback. `nonSurfaceStop` reports that the descent
// ended because the point lies in a Solid/Empty/absent child, in which case the
// caller must not treat the parent as a surface ring vertex (findCellAt would
// have returned that non-surface cell and isSurfaceCell would have rejected it).
uint rmDescendFrom(uint idx, vec3 pos, out bool nonSurfaceStop) {
    nonSurfaceStop = false;
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

bool rmNodeContains(uint id, vec3 pos) {
    vec3 mn = rmNodeMin(id);
    return all(greaterThanEqual(pos, mn)) &&
           all(lessThanEqual(pos, mn + vec3(rmNodeSize(id))));
}

// Half-open membership matches the >= child-slot rule: a point on a shared
// face belongs to the positive child. Used when deciding whether a stacked
// ancestor is a safe starting point, so an exact boundary point cannot start
// in the sibling the root descent would not choose.
bool rmNodeContainsInterior(uint id, vec3 pos) {
    vec3 mn = rmNodeMin(id);
    return all(greaterThanEqual(pos, mn)) &&
           all(lessThan(pos, mn + vec3(rmNodeSize(id))));
}

uint rmFindNodeAtImpl(uint root, vec3 pos, out bool nonSurfaceStop) {
    nonSurfaceStop = false;
    if (!rmNodeContains(root, pos)) return RM_INVALID_NODE;
    return rmDescendFrom(root, pos, nonSurfaceStop);
}

uint rmFindNodeAt(uint root, vec3 pos, out bool nonSurfaceStop) {
    rmCount(RM_COUNTER_RING_DESCENTS);
    return rmFindNodeAtImpl(root, pos, nonSurfaceStop);
}

// Phase 1 ancestor-start descent. The nearest accepted hit's node and its
// copied ancestor chain are safe starting points when they contain `pos`: the
// child choices below a containing node depend only on `pos`, so skipping the
// ancestors cannot change the result. Fall back to the layer root otherwise.
uint rmFindNodeFromStack(uint root, uint hitNode, vec3 pos, out bool nonSurfaceStop) {
    if (hitNode != RM_INVALID_NODE && rmNodeContainsInterior(hitNode, pos))
        return rmDescendFrom(hitNode, pos, nonSurfaceStop);
    for (int d = rmHitStackDepth - 1; d >= 0; --d) {
        uint n = rmHitStackNode[d];
        if (rmNodeContainsInterior(n, pos)) return rmDescendFrom(n, pos, nonSurfaceStop);
    }
    // Only a fallback that restarts at the layer root is a "ring descent" in
    // the counter sense; ancestor starts are the saving Phase 1 measures.
    rmCount(RM_COUNTER_RING_DESCENTS);
    return rmFindNodeAtImpl(root, pos, nonSurfaceStop);
}

// EdgeCell::isSurface() + nodeTopologyAllowed() from OctreeTriangles.comp: a
// ring vertex must be a Surface node whose own corners cross zero and whose
// paint is not a raw discard. (Infinity corners are skipped by sdfEvalType.)
// meta.y is the child base now, not a SpaceType: the table only stores Surface
// nodes, so the old type check is implied by construction.
bool rmBlendCellValid(uint id) {
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
    b.normal = vec3(0.0); // caller keeps the march's analytic normal
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
    vec3 ringNormal[4];
    int ringMat[4];
    int ringType[4];
    vec2 ringUv[4];
    vec3 ringHsv[4];
    int ringCount = 0;
    float dedupEps = max(sz * 1e-4, 1e-6);
    for (int q = 0; q < 4; ++q) {
        rmCount(RM_COUNTER_RING_CELLS);
        int su, sv;
        rmQuadrantSigns(bestAxis, q, su, sv);
        vec3 sp = vec3(0.0);
        sp[bestAxis] = tSample;
        sp[bestU] = bestFixedU + (su < 0 ? -offset : offset);
        sp[bestV] = bestFixedV + (sv < 0 ? -offset : offset);

        bool nonSurfaceStop;
        uint id = rmFindNodeFromStack(root, hitId, sp, nonSurfaceStop);
        if (nonSurfaceStop || id == RM_INVALID_NODE || !rmBlendCellValid(id)) continue;
        vec3 qp = rmCellQef(id);
        bool duplicate = false;
        for (int k = 0; k < ringCount; ++k) {
            vec3 dq = qp - ringBase[k];
            if (ringId[k] == id || dot(dq, dq) <= dedupEps * dedupEps) {
                rmCount(RM_COUNTER_RING_DUPLICATES);
                duplicate = true;
                break;
            }
        }
        if (!duplicate && ringCount < 4) {
            int m = rmRingMaterial(id, qp);
            vec3 n = rmNodeNormalAtQef(id, qp);
            float disp = rmDisplacementN(id, m, qp, n);
            ringId[ringCount] = id;
            ringBase[ringCount] = qp;
            ringPos[ringCount] = qp + n * disp;
            ringNormal[ringCount] = n;
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
        uint rid[4]; vec3 rbase[4]; vec3 rpos[4]; vec3 rnorm[4];
        int rmat[4]; int rtype[4]; vec2 ruv[4]; vec3 rhsv[4];
        for (int i = 0; i < ringCount; ++i) {
            int k = (i + owner) % ringCount;
            rid[i] = ringId[k];
            rbase[i] = ringBase[k];
            rpos[i] = ringPos[k];
            rnorm[i] = ringNormal[k];
            rmat[i] = ringMat[k];
            rtype[i] = ringType[k];
            ruv[i] = ringUv[k];
            rhsv[i] = ringHsv[k];
        }
        for (int i = 0; i < ringCount; ++i) {
            ringId[i] = rid[i];
            ringBase[i] = rbase[i];
            ringPos[i] = rpos[i];
            ringNormal[i] = rnorm[i];
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
    // Raster fragNormal is the barycentric interpolation of the per-vertex
    // cell normals; blend the same corner normals here.
    outBlend.normal = ringNormal[ci0] * w.x + ringNormal[ci1] * w.y + ringNormal[ci2] * w.z;
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

// March one layer root, stopping at tMax. Adaptive descent: the ray is
// confined to the current cell's [t, nodeExit] interval and follows stored
// Surface children. Empty and Solid children are skipped (neither can contain
// an isosurface crossing); only node-cap-pruned Surface children fall back to
// the parent trilinear field. Within a cell the field is sphere-traced with the
// node's cached 1/Lipschitz bound (0.5 safety factor when no payload is
// available) and a sign-change bisection refines the crossing. Discarded
// (underground / -Y NormalBrush) surfaces are skipped and the march continues.
// tMax is the nearest hit already found by an earlier root: it clamps the root
// box exit and aborts the loop as soon as the ray reaches it, so a root ordered
// behind a hit costs only its AABB test. Exact: the caller only passes a hit
// distance produced by the same field, and a first crossing at or beyond it can
// never be the nearest.
// Hit acceptance (raster-parity visibility rules).
//   Solid: back-face culling (the raster solid pipeline culls back faces) and
//          DISCARD_BRUSH_INDEX cells are not surfaces at all.
//   Water: the NormalBrush layer defines a water volume, so BOTH the top and
//          bottom water faces are visible (the raster water back-face pass
//          renders the underside when the camera is underwater). Accept
//          horizontal faces with either sign of n.y; reject the near-vertical
//          volume sides, which are the NormalBrush discards that used to
//          appear as water walls.
bool rmHitAllowed(uint node, vec3 p, vec3 rd, int material, bool waterRoot,
                  out vec3 normal) {
    // rmNodeNormal is already normalized; keep that single-normalization value
    // for the consumers that previously used it directly (displacement base
    // step and ring fallback). The visibility tests retain the historical
    // extra normalize so their last-bit behavior is unchanged.
    normal = rmNodeNormal(node, p);
    if (waterRoot) return abs(normalize(normal).y) > 1.0e-3;
    if (material == RAYMARCH_DISCARD_BRUSH) return false;
    return dot(normalize(normal), rd) < 0.0;
}

bool rmMarchRoot(uint root, vec3 ro, vec3 rd, vec3 invRd,
                 float tMax, bool haveRootInterval, float rootT0, float rootT1,
                 out uint hitNode, out float hitT, out int hitBrush,
                 out vec3 hitNormal, bool waterRoot) {
    hitNormal = vec3(0.0);
    float t0, t1;
    if (haveRootInterval) {
        // The caller already tested the root cube with the same invRd (the
        // tessellation-displacement rejection box is the root cube).
        t0 = rootT0;
        t1 = rootT1;
    } else if (!rmRayBoxInv(rmNodeMin(root), rmNodeSize(root), ro, invRd, t0, t1)) {
        rmCount(RM_COUNTER_EARLY_ROOTS);
        return false;
    }
    t1 = min(t1, tMax);
    float t = max(t0, 0.0);
    if (t >= t1) {
        rmCount(RM_COUNTER_EARLY_ROOTS);
        return false;
    }
    rmCount(RM_COUNTER_ROOT_MARCHES);
    rmCount(RM_COUNTER_NODE_VISITS);

    uint stackNode[RM_MAX_STACK];
    float stackExit[RM_MAX_STACK];
    int depth = 0;

    uint node = root;
    float nodeExit = t1;

    // Carry-forward sample. The loop evaluates the field at t and then at
    // t+dt; on the next iteration it would evaluate the same node at the same
    // t again. Reuse that exact value when neither the node nor the parameter
    // changed. The cache is invocation-local registers, so it needs no
    // invalidation beyond the equality test.
    uint cachedNode = 0xFFFFFFFFu;
    float cachedT = -1.0;
    float cachedD = 0.0;
    float cachedBase = 0.0;
    float cachedMaxDisp = 0.0;

    // Previous accepted sample of the CURRENT node, for the secant refinement
    // of a hit accepted inside a screen-space-widened tolerance band. Reset on
    // every discard path so a stale bracket is never reused.
    uint prevNode = 0xFFFFFFFFu;
    float prevT = -1.0;
    float prevD = 0.0;

    for (int iter = 0; iter < RM_MAX_STEPS; ++iter) {
        // An earlier root already found a nearer hit: this march cannot
        // improve it and every candidate from here on is behind tMax.
        if (t >= tMax) return false;
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
        if (rmNodeIsLeaf(node)) rmCount(RM_COUNTER_LEAF_ITERS);
        else rmCount(RM_COUNTER_NONLEAF_ITERS);

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
            if (rmRayBoxInv(childMin, childSize, pn, invRd, c0, c1) && c1 > 0.0) {
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
                    rmCount(RM_COUNTER_NODE_VISITS);
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
        float d;
        if (node == cachedNode && t == cachedT) {
            // Exact reuse: same node, same parameter, same expression.
            rmCount(RM_COUNTER_CARRY_HITS);
            d = cachedD;
            baseD = cachedBase;
            maxDisp = cachedMaxDisp;
        } else {
            rmCount(RM_COUNTER_FIELD_EVALS);
            d = rmField(node, p, material, baseD, maxDisp);
        }
        if (isnan(d)) d = 1.0; // defensive: never let a bad sample freeze the march
        float eps = rmHitEpsilon(nodeSize, t);

        if (abs(d) <= eps) {
            // Screen-space tolerance: the accepted sample can sit up to `eps`
            // (a few pixels at distance) before the true crossing. Refine with
            // the secant of the two same-node samples that bracket it, clamped
            // to the band.
            float tHit = t;
            if (ubo.raymarchScreenEps > 0.0 && prevNode == node)
                tHit = rmRefineHit(node, ro, rd, material, prevT, prevD, t, d, eps);
            // Raster-parity visibility: solid back faces and DISCARD_BRUSH_INDEX
            // cells are not shaded; water accepts only the +Y surface (top
            // plane, from either side), not the volume sides/bottom.
            vec3 candidateNormal;
            if (rmHitAllowed(node, ro + rd * tHit, rd, material, waterRoot,
                             candidateNormal)) {
                hitNode = node;
                hitT = tHit;
                hitBrush = material;
                hitNormal = candidateNormal;
                // Preserve this hit's ancestor chain for the hit path; later
                // root marches only overwrite it when they find a nearer hit.
                rmHitStackDepth = depth;
                for (int i = 0; i < depth; ++i) rmHitStackNode[i] = stackNode[i];
                return true;
            }
            rmCount(RM_COUNTER_DISCARD_SKIPS);
            t += eps;
            prevNode = 0xFFFFFFFFu;
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
            // Payload cache: 1/L for the node's conservative trilinear
            // Lipschitz bound (OctreeDerive.comp), so smooth cells take steps
            // up to 1.155x longer and rough cells step shorter (safer) than
            // the fixed 0.5 factor. Cells without a payload keep 0.5.
            float invLip = ((uint(rmNodes[node].meta.w) & RM_FLAG_PAYLOAD) != 0u)
                ? rmNodes[node].hsv1.z
                : 0.5;
            stepLen = max(abs(d) * invLip, eps);
        }
        float tNext = min(t + stepLen, cellExit);
        float dNextBase;
        float dNextMax;
        rmCount(RM_COUNTER_FIELD_EVALS);
        float dNext = rmField(node, ro + rd * tNext, material, dNextBase, dNextMax);
        if (isnan(dNext)) {
            // Cache only finite values: the displacement branch consumes
            // baseD/maxDisp too, and a NaN there would poison the next reuse.
            dNext = -1.0;
            dNextBase = -1.0;
            dNextMax = 0.0;
        }

        // Sign change inside this cell: refine with bisection.
        if (d > 0.0 && dNext <= 0.0) {
            float lo = t;
            float hi = tNext;
            for (int b = 0; b < RM_BISECT_STEPS; ++b) {
                float mid = 0.5 * (lo + hi);
                rmCount(RM_COUNTER_BISECT_EVALS);
                float bBase;
                float bMax;
                float dm = rmField(node, ro + rd * mid, material, bBase, bMax);
                if (isnan(dm)) break;
                if (dm > 0.0) lo = mid; else hi = mid;
            }
            float tHit = 0.5 * (lo + hi);
            vec3 candidateNormal;
            if (rmHitAllowed(node, ro + rd * tHit, rd, material, waterRoot,
                             candidateNormal)) {
                hitNode = node;
                hitT = tHit;
                hitBrush = material;
                hitNormal = candidateNormal;
                rmHitStackDepth = depth;
                for (int i = 0; i < depth; ++i) rmHitStackNode[i] = stackNode[i];
                return true;
            }
            rmCount(RM_COUNTER_DISCARD_SKIPS);
            t = tHit + eps;
            prevNode = 0xFFFFFFFFu;
            continue;
        }

        cachedNode = node;
        cachedT = tNext;
        cachedD = dNext;
        cachedBase = dNextBase;
        cachedMaxDisp = dNextMax;
        // Bracket endpoint for the next iteration's hit refinement.
        prevNode = node;
        prevT = t;
        prevD = d;
        t = tNext;
    }
    return false;
}

// ── Hit shading payload (primary surface and underwater bed) ──────────────
// Reconstruct the same undisplaced base point / material blend / normal the
// raster TES + TCS produce for the triangle under a marched hit. Shared by the
// primary hit and by the lake bed traced for water refraction.
struct RmHitShading {
    vec3 basePos;
    vec3 normal;
    RmSurfaceBlend blend;
    uint blendCell;
};

RmHitShading rmResolveHitShading(uint rootNode, uint node, int material,
                                 vec3 hitPos, vec3 rd, vec3 acceptedNormal) {
    RmHitShading h;
    // Tessellation parity: the raster TES samples the height maps at the
    // UNDISPLACED vertex and moves it along the smooth cell normal. Recover
    // that base point (and its normal) from the bumped hit so shading and
    // mapping are evaluated at the same world position the raster path uses.
    // Two fixed-point steps; the offset is small relative to the height-map
    // feature scale, so this converges to the raster's undisplaced vertex.
    // With tessellation off (the default) no displacement exists, so the
    // hit-point gradient is not solved again: rmHitAllowed already computed
    // the accepted normal at this exact point. With displacement active, that
    // same normal feeds the base-point step and the height-map sample instead
    // of being solved twice. A zero acceptedNormal is only a safety fallback
    // for callers that predate the return value.
    vec3 sdfNormal = acceptedNormal;
    if (dot(sdfNormal, sdfNormal) <= 1e-12) sdfNormal = rmNodeNormal(node, hitPos);
    float hitDisplacement = 0.0;
    if (rmDisplacementActive(material)) {
        hitDisplacement = rmDisplacement(node, material, hitPos, sdfNormal);
    }
    h.basePos = hitPos;
    if (hitDisplacement != 0.0) {
        h.basePos = hitPos - sdfNormal * hitDisplacement;
        for (int i = 0; i < 2 && hitDisplacement != 0.0; ++i) {
            sdfNormal = rmNodeNormal(node, h.basePos);
            hitDisplacement = rmDisplacement(node, material, h.basePos, sdfNormal);
            h.basePos = hitPos - sdfNormal * hitDisplacement;
        }
    }

    // The raster material blend belongs to the base (undisplaced) cell: use
    // that cell when it is in the table, otherwise the bumped hit's own cell.
    bool baseCellNonSurface;
    uint blendCell = rmFindNodeFromStack(rootNode, node, h.basePos, baseCellNonSurface);
    if (blendCell == RM_INVALID_NODE || baseCellNonSurface) blendCell = node;
    h.blend = rmBlendAtHit(rootNode, blendCell, h.basePos, hitPos);

    // Raster fragNormal is the barycentric interpolation of the per-vertex
    // cell normals; use the blend's normal when the ring reconstructed it and
    // fall back to the marched cell's analytic normal otherwise. The fallback
    // is only solved when the ring did not provide one, so the common
    // tessellation-off hit pays no hit-point gradient at all.
    vec3 fallbackNormal = sdfNormal;
    if (dot(fallbackNormal, fallbackNormal) <= 1e-12) {
        fallbackNormal = rmNodeNormal(node, hitPos);
    }
    h.normal = (dot(h.blend.normal, h.blend.normal) > 1e-12)
        ? normalize(h.blend.normal) : fallbackNormal;
    if (dot(h.normal, rd) > 0.0) h.normal = -h.normal; // front surface only
    h.blendCell = blendCell;
    return h;
}

// Fill the fragment globals shadeSolidSurface() reads. gl_FragDepth is NOT
// touched here: the primary hit owns the depth of the water/solid surface,
// while a secondary (underwater bed) shading pass must not overwrite it.
void rmWriteFragGlobals(vec3 hitPos, RmHitShading h) {
    fragPosWorld = hitPos;
    fragPosWorldNotDisplaced = h.basePos;
    fragNormal = h.normal;
    fragSharpNormal = h.normal;
    fragTexIndices = h.blend.indices;
    fragTextureTypes = h.blend.types;
    fragTexWeights = h.blend.weights;
    fragHSV = h.blend.hsv;
    // Interpolated per-vertex UV scale (OctreeNode::data), matching the
    // raster's interpolated fragUV.
    fragUV = h.blend.uv;
    fragPosLightSpace = ubo.lightSpaceMatrix * vec4(hitPos, 1.0);
    // Same feed the TES writes (tess levels / 16, clamped to the heat range).
    // The raster's tess levels are per base cell, not per bumped hit.
    fragTessLevel = vec3(rmTessLevel(h.blendCell, rmCachedBrush(h.blendCell), h.basePos));
}

// Front-to-back root scheduling shared by the primary march and the water
// lobes. Each entry is the root's ray interval entry parameter; sorting by
// entry lets a root behind an already-found hit be skipped without marching
// it. The rejection box (see rmRootTestBox) is a conservative bound on every
// point the root march can sample, so a missed or behind-origin box cannot
// hide a crossing.
struct RmRootInterval {
    uint root;
    float t0;
    float t1;
};

// Stable insertion into the ascending-t0 list: roots with the same entry keep
// their table order, so exact ties select the same root as the old unsorted
// loop. t1 is retained so the march can reuse the caller's root-cube slab
// result when the rejection box is the root cube (tessellation active).
void rmInsertRootInterval(inout RmRootInterval iv[RM_MAX_ROOTS],
                          inout int count, uint root, float t0, float t1) {
    int j = count;
    while (j > 0 && iv[j - 1].t0 > t0) {
        iv[j] = iv[j - 1];
        --j;
    }
    iv[j].root = root;
    iv[j].t0 = t0;
    iv[j].t1 = t1;
    ++count;
}

// Root rejection box. The stored-cell surface AABB bounds the unbumped
// isosurface exactly, but tessellation displacement folds an offset into the
// field and can move the surface outside those cells. While displacement is
// enabled the only safe rejection box is the root cube itself, which is the
// interval the march actually samples anyway.
void rmRootTestBox(uint root, out vec3 mn, out vec3 mx) {
    if (ubo.tessellationEnabled == 0u) {
        mn = rmRoots[root].surfaceMin.xyz;
        mx = rmRoots[root].surfaceMax.xyz;
    } else {
        mn = rmNodeMin(rmRoots[root].node);
        mx = mn + vec3(rmNodeSize(rmRoots[root].node));
    }
}

// Nearest hit among the SOLID layer roots only. The water refraction path uses
// it to reach the lake bed behind the water surface (the primary march stops
// at the water layer).
bool rmMarchSolidRoots(vec3 ro, vec3 rd, out uint hitRoot, out uint hitNode,
                       out float hitT, out int hitBrush, out vec3 hitNormal) {
    rmCount(RM_COUNTER_WATER_SECONDARY);
    hitT = 1.0e30;
    hitRoot = 0u;
    hitNode = 0u;
    hitBrush = RAYMARCH_DISCARD_BRUSH;
    hitNormal = vec3(0.0);
    bool found = false;
    // Hoist the reciprocals once: every root AABB test below reuses them.
    vec3 invRd = vec3(rmInvComp(rd.x), rmInvComp(rd.y), rmInvComp(rd.z));

    RmRootInterval iv[RM_MAX_ROOTS];
    int ivCount = 0;
    const uint sortedRoots = min(pc.rootCount, uint(RM_MAX_ROOTS));
    for (uint i = 0u; i < sortedRoots; ++i) {
        if (rmRoots[i].kind != 0u) continue;
        vec3 testMin, testMax;
        rmRootTestBox(i, testMin, testMax);
        float t0, t1;
        rmCount(RM_COUNTER_ROOT_TESTS);
        if (!rmRayBoxInv(testMin, testMax - testMin, ro, invRd, t0, t1)) {
            rmCount(RM_COUNTER_EARLY_ROOTS);
            continue;
        }
        rmInsertRootInterval(iv, ivCount, i, t0, t1);
    }

    // Front to back: the first root whose entry is at or behind the nearest
    // hit cannot contribute, and every march stops at that hit.
    for (int k = 0; k < ivCount; ++k) {
        if (iv[k].t0 >= hitT) {
            rmCount(RM_COUNTER_EARLY_ROOTS);
            break;
        }
        const uint i = iv[k].root;
        uint node;
        float t;
        int brush;
        vec3 normal;
        if (rmMarchRoot(rmRoots[i].node, ro, rd, invRd, hitT,
                        ubo.tessellationEnabled != 0u, iv[k].t0, iv[k].t1,
                        node, t, brush, normal, false) && t < hitT) {
            hitT = t;
            hitNode = node;
            hitRoot = i;
            hitBrush = brush;
            hitNormal = normal;
            found = true;
        }
    }

    // Roots past the sorting budget keep the old unsorted order (still with
    // the surface-AABB rejection and the nearest-hit clamp).
    for (uint i = sortedRoots; i < pc.rootCount; ++i) {
        if (rmRoots[i].kind != 0u) continue;
        vec3 testMin, testMax;
        rmRootTestBox(i, testMin, testMax);
        float t0, t1;
        rmCount(RM_COUNTER_ROOT_TESTS);
        if (!rmRayBoxInv(testMin, testMax - testMin, ro, invRd, t0, t1)) {
            rmCount(RM_COUNTER_EARLY_ROOTS);
            continue;
        }
        uint node;
        float t;
        int brush;
        vec3 normal;
        if (rmMarchRoot(rmRoots[i].node, ro, rd, invRd, hitT,
                        ubo.tessellationEnabled != 0u, t0, t1,
                        node, t, brush, normal, false) && t < hitT) {
            hitT = t;
            hitNode = node;
            hitRoot = i;
            hitBrush = brush;
            hitNormal = normal;
            found = true;
        }
    }
    return found;
}

// ── Ray-marched water surface waves ───────────────────────────────────────
// The SDF water surface is flat, so the raster water's animated swell is
// recovered per pixel exactly like its fragment normal: the shared fBm wind
// field sets the propagation direction and the saturating wind-sea amplitude,
// and the analytic Gerstner gradient tilts the SDF normal. The same sample
// returns the vertical wave height so the reflection/refraction origins ride
// the moving surface.
struct RmWaterSurface {
    vec3 normal;  // wave-perturbed interface normal
    float height; // vertical displacement along the flat SDF normal (m)
};

RmWaterSurface rmWaterSurface(WaterParamsNamed wp, vec3 p, vec3 N, float detail,
                               vec2 dPosdx, vec2 dPosdy) {
    RmWaterSurface s;
    s.normal = N;
    s.height = 0.0;

    // Shared animation clock (SkyUniform::windTime is the live main clock,
    // advanced every frame even when the raster water pass is skipped).
    float t = sky.windTime;
    vec3 w = windSVF(p, t);
    vec2 windXZ = w.xz;
    float windSpeed = length(windXZ);
    // Propagation direction: the live wind; the authored angle is only the
    // still-air fallback (same rule as waterWindWave() in WaterTese.glsl).
    vec2 dir = (windSpeed > 1e-4) ? windXZ / windSpeed
                                  : normalize(wp.waveDirection + vec2(1e-5, 0.0));

    // The raster surface applies the per-pixel FBM ripple normal independently
    // of the Gerstner swell (sub-tessellation detail), so the ray-marched
    // surface does the same. Without it a hit with waves off falls back to the
    // perfectly flat SDF normal and reads as flat shading.
    if (!wp.enableWaves || wp.bumpAmplitude <= 0.0) {
        s.normal = waterRippleNormal(N, N, p, dir, t, wp, detail);
        return s;
    }

    // Saturating wind response: calm air leaves nearly flat water, strong wind
    // builds the authored sea. response = 0 keeps the height constant.
    float ref = max(wp.windWaveReference, 0.5);
    float windNorm = 1.0 - exp(-max(windSpeed, 0.0) / ref);
    float ampScale = mix(1.0, windNorm, clamp(wp.windWaveResponse, 0.0, 1.0));

    // Deep-water sentinel: the SDF layer does not store a column depth here.
    // Using the shore fade would flatten the whole lake; the waterline case is
    // already handled by the near-zero water thickness at the shore.
    WaterWaveField wf = waterWaveField(p, t, -1.0,
        wp.bumpAmplitude * ampScale, dir, wp, false, 3,
        dPosdx, dPosdy);
    if (wf.calmMask <= 0.0) {
        s.normal = waterRippleNormal(N, N, p, dir, t, wp, detail);
        return s;
    }

    vec3 up = abs(N.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T = normalize(cross(up, N));
    vec3 B = cross(N, T);
    vec3 n = normalize(N - dot(wf.grad, T) * T - dot(wf.grad, B) * B);
    if (!(dot(n, n) > 0.25)) n = N;
    if (dot(n, N) < 0.0) n = -n;
    s.normal = waterRippleNormal(n, N, p, dir, t, wp, detail);
    s.height = wf.height;
    return s;
}

// Shade one raymarched solid hit: reconstruct the material payload, fill the
// shared fragment globals and run the raster solid shading body. Returns the
// shaded color (outColor is the target shadeSolidSurface writes).
vec3 rmShadeSolidHit(uint root, uint node, int brush, vec3 pos, vec3 rd,
                     vec3 acceptedNormal) {
    RmHitShading h = rmResolveHitShading(rmRoots[root].node, node,
                                         max(brush, 0), pos, rd, acceptedNormal);
    rmWriteFragGlobals(pos, h);
    shadeSolidSurface(vec3(0.0));
    return outColor.rgb;
}

// Water shading for a raymarched water-layer hit.
//
// The octree raymarch draws water inside the SOLID pass (opaque color+depth),
// so the raster water renderer's own targets, its separate underwater pass and
// the previous-frame screen-space sources do not exist for this surface.
// Both lobes are therefore traced through the SAME SDF octree the primary ray
// used (no SSR / no previous-frame color dependency):
//   * refraction: the Snell-bent ray continues through the water to the SOLID
//     lake bed, which is shaded with the raster path's shadeSolidSurface() and
//     attenuated by the water column (absorption + tint);
//   * reflection: a ray from the wave-displaced surface along the mirror
//     direction marches the solid roots and shades the hit, with procedural
//     sky + clouds as the miss value;
//   * the swell (wind-driven Gerstner slope) and the per-pixel FBM ripple
//     detail (shared with the raster surface) perturb the normal and move the
//     ray origins, so both images animate with the live wind;
//   * the two are composed with the raster surface's exact Fresnel/strength
//     mirror mix, and the CSM surface shadow darkens the result.
// The RT variant can resolve the exact scene instead (see rtResolveWaterHit).
vec3 shadeRaymarchWaterSDF(WaterParamsNamed wp, vec3 hitPos, vec3 shadePos,
                           vec3 N, vec3 Nw, vec3 V) {
    vec3 toSun = normalize(-ubo.lightDirection);
    vec3 rd = -V;
    float ndv = clamp(dot(Nw, V), 0.0, 1.0);
    // Raster water Fresnel: Schlick base + refraction/reflection mix.
    float fresnel = clamp(0.02 + 0.98 * pow(1.0 - ndv,
                          clamp(wp.fresnelPower, 1.0, 8.0)), 0.0, 1.0);

    // CSM shadow on the water surface (the refracted bed carries its own).
    float surfaceShadow = 0.0;
    if (ubo.shadowsEnabled != 0u) {
        float ndlS = max(dot(Nw, toSun), 0.0);
        if (ndlS > 0.01) {
            float bias = max(0.002 * (1.0 - ndlS), 0.0005);
            surfaceShadow = ShadowCalculation(
                ubo.lightSpaceMatrix * vec4(hitPos, 1.0), hitPos, bias);
        }
    }

    // ── Refraction: Snell-bent ray traced through the SDF to the bed ──
    vec3 snellDir = refract(rd, Nw, 1.0 / max(wp.waterIor, 1.0));
    if (dot(snellDir, snellDir) < 1e-6) snellDir = reflect(rd, Nw); // total internal
    snellDir = normalize(snellDir);
    // Refraction Strength = amount of bending (0 = straight through).
    vec3 refrDir = normalize(mix(rd, snellDir,
                                 clamp(wp.refractionStrength, 0.0, 1.0)));
    if (!wp.enableRefraction) refrDir = rd;

    float thickness = max(wp.maxThickness, 0.0);
    vec3 refrCol;
    // The refracted ray starts just below the flat SDF surface: the swell is a
    // shading/rotational deformation here, not a geometric displacement of the
    // marched plane, so the origin must stay on the real interface.
    vec3 refrOrigin = hitPos - N * 0.02; // just inside the water volume
    uint bedRoot, bedNode;
    float bedT;
    int bedBrush;
    vec3 bedNormal;
    if (rmMarchSolidRoots(refrOrigin, refrDir, bedRoot, bedNode, bedT, bedBrush,
                          bedNormal)) {
        thickness = min(bedT, max(wp.maxThickness, 0.0));
        vec3 bedPos = refrOrigin + refrDir * bedT;
        // Shade the bed with the shared solid shading body (materials,
        // triplanar mapping, displacement parity, CSM shadows). It writes
        // outColor; the water composition below replaces it.
        vec3 bedColor = rmShadeSolidHit(bedRoot, bedNode, bedBrush, bedPos, refrDir,
                                        bedNormal);
        // Beer-Lambert absorption over the water column, then the same tint
        // ramp the raster surface uses (shore fade + transparency cap).
        vec3 transmittance = exp(-min(
            wp.absorption * max(thickness * wp.absorptionScale, 0.0),
            vec3(2.5)));
        float depthFade = 1.0 - exp(-thickness * max(wp.depthFalloff, 1e-4));
        if (wp.tintShoreFadeDepth > 0.0 && thickness > 1e-4) {
            depthFade *= smoothstep(0.0, max(wp.tintShoreFadeDepth, 1e-4), thickness);
        }
        float tintBlend = clamp(depthFade * wp.waterTint, 0.0,
                                clamp(1.0 - wp.transparency, 0.0, 1.0));
        refrCol = mix(bedColor * transmittance, wp.waterColor, tintBlend);
    } else {
        // No bed within the layer: deep water, sky along the refracted ray
        // saturated to the deep tint (raster deep-water continuity).
        vec3 deep = rtProceduralSky(refrDir, sky.horizonColor,
                                    sky.zenithColor, sky.exponent);
        refrCol = mix(deep, wp.waterColor, 0.85);
    }

    // ── Reflection: ray march the SDF scene along the mirror direction ──
    vec3 reflCol = vec3(0.0);
    if (wp.enableReflection) {
        vec3 reflDir = normalize(reflect(rd, Nw));
        // Water is a height field: a microfacet never mirrors below the
        // horizon, and a downward secondary ray would re-hit the lake bed.
        reflDir = (length(reflDir.xz) > 1e-6 || reflDir.y > 1e-6)
            ? normalize(vec3(reflDir.x, max(reflDir.y, 0.0), reflDir.z))
            : Nw;
        uint rRoot, rNode;
        float rT;
        int rBrush;
        vec3 reflNormal;
        vec3 reflOrigin = shadePos + Nw * 0.05;
        if (rmMarchSolidRoots(reflOrigin, reflDir, rRoot, rNode, rT, rBrush,
                              reflNormal)) {
            vec3 rPos = reflOrigin + reflDir * rT;
            reflCol = rmShadeSolidHit(rRoot, rNode, rBrush, rPos, reflDir,
                                      reflNormal);
        } else {
            reflCol = rtProceduralSky(reflDir, sky.horizonColor,
                                      sky.zenithColor, sky.exponent);
            if (sky.cloudsEnabled != 0u) {
                float day = smoothstep(-0.2, 0.2, -ubo.lightElevation);
                reflCol += cloudApproxForReflection(reflDir, toSun, day);
            }
        }
    }

    // ── Compose exactly like the raster water surface ──
    // Reflection Strength is the mirror amount (1 = full mirror, 0 = physical
    // Fresnel only), matching WaterSurface.glsl's reflMix.
    vec3 color = refrCol;
    if (wp.enableReflection) {
        float reflMix = wp.uniformReflection
            ? clamp(wp.reflectionStrength, 0.0, 1.0)
            : mix(fresnel, 1.0, clamp(wp.reflectionStrength, 0.0, 1.0));
        color = mix(refrCol, reflCol, reflMix);
    }
    return color * mix(1.0, 0.55, surfaceShadow);
}

void main() {
    rmCount(RM_COUNTER_RAYS);
    rmHitStackDepth = 0;
    vec2 extent = max(pc.viewport.xy, vec2(1.0));
    vec2 ndc = gl_FragCoord.xy / extent * 2.0 - 1.0;

    vec4 farPoint = ubo.invViewProjection * vec4(ndc, 1.0, 1.0);
    if (abs(farPoint.w) < 1e-8) discard;

    vec3 ro = ubo.viewPosition;
    vec3 rd = normalize(farPoint.xyz / farPoint.w - ro);

    // Nearest hit across every visible terrain layer root (Solid + Water).
    // Roots are ordered by rejection-box entry (the stored-cell surface AABB,
    // or the root cube while tessellation displacement is active), so the first
    // hit found clamps every later march. The box bounds every point the root
    // march samples, so rejecting a missed root or stopping behind a hit cannot
    // skip geometry.
    float nearestT = 1.0e30;
    uint nearestNode = 0u;
    uint nearestRoot = 0u;
    int nearestBrush = RAYMARCH_DISCARD_BRUSH;
    vec3 nearestNormal = vec3(0.0);
    bool found = false;
    vec3 invRd = vec3(rmInvComp(rd.x), rmInvComp(rd.y), rmInvComp(rd.z));

    RmRootInterval iv[RM_MAX_ROOTS];
    int ivCount = 0;
    const uint sortedRoots = min(pc.rootCount, uint(RM_MAX_ROOTS));
    for (uint i = 0u; i < sortedRoots; ++i) {
        vec3 testMin, testMax;
        rmRootTestBox(i, testMin, testMax);
        float t0, t1;
        rmCount(RM_COUNTER_ROOT_TESTS);
        if (!rmRayBoxInv(testMin, testMax - testMin, ro, invRd, t0, t1)) {
            rmCount(RM_COUNTER_EARLY_ROOTS);
            continue;
        }
        rmInsertRootInterval(iv, ivCount, i, t0, t1);
    }

    for (int k = 0; k < ivCount; ++k) {
        if (iv[k].t0 >= nearestT) {
            rmCount(RM_COUNTER_EARLY_ROOTS);
            break;
        }
        const uint i = iv[k].root;
        uint node;
        float t;
        int brush;
        vec3 normal;
        if (rmMarchRoot(rmRoots[i].node, ro, rd, invRd, nearestT,
                        ubo.tessellationEnabled != 0u, iv[k].t0, iv[k].t1,
                        node, t, brush, normal, rmRoots[i].kind == 1u) &&
            t < nearestT) {
            nearestT = t;
            nearestNode = node;
            nearestRoot = i;
            nearestBrush = brush;
            nearestNormal = normal;
            found = true;
        }
    }

    // Roots past the sorting budget keep the old unsorted order (still with
    // the surface-AABB rejection and the nearest-hit clamp).
    for (uint i = sortedRoots; i < pc.rootCount; ++i) {
        vec3 testMin, testMax;
        rmRootTestBox(i, testMin, testMax);
        float t0, t1;
        rmCount(RM_COUNTER_ROOT_TESTS);
        if (!rmRayBoxInv(testMin, testMax - testMin, ro, invRd, t0, t1)) {
            rmCount(RM_COUNTER_EARLY_ROOTS);
            continue;
        }
        uint node;
        float t;
        int brush;
        vec3 normal;
        if (rmMarchRoot(rmRoots[i].node, ro, rd, invRd, nearestT,
                        ubo.tessellationEnabled != 0u, t0, t1,
                        node, t, brush, normal, rmRoots[i].kind == 1u) &&
            t < nearestT) {
            nearestT = t;
            nearestNode = node;
            nearestRoot = i;
            nearestBrush = brush;
            nearestNormal = normal;
            found = true;
        }
    }
    if (!found) discard;
    rmCount(RM_COUNTER_HITS);
    const bool isWater = rmRoots[nearestRoot].kind == 1u;
    // rmHitAllowed() already rejected solid DISCARD_BRUSH_INDEX cells and
    // non-upward water faces. Keep the solid guard as a defensive backstop;
    // water shading is driven by the +Y surface geometry, not the material
    // index (shore cells may still carry the NormalBrush discard value).
    if (!isWater && nearestBrush == RAYMARCH_DISCARD_BRUSH) discard;

    vec3 hitPos = ro + rd * nearestT;

    // Water hits use the analytic SDF normal directly: a water node is not a
    // terrain material cell, so the tessellation displacement and brush-blend
    // machinery (which follows a terrain height map) must not perturb its
    // interface normal. Keeping the water normal clean is what makes the
    // reflection/refraction directions stable pixel-to-pixel.
    vec3 normal;
    if (isWater) {
        // Reuse the normal rmHitAllowed already solved at this exact hit.
        vec3 sdfNormal = (dot(nearestNormal, nearestNormal) > 1e-12)
            ? normalize(nearestNormal)
            : normalize(rmNodeNormal(nearestNode, hitPos));
        if (dot(sdfNormal, rd) > 0.0) sdfNormal = -sdfNormal;

        // The analytic gradient above is the TRILINEAR cell gradient, so it
        // jumps at every cell boundary and flat-shades curved water bodies
        // (the transparent spheres) into large facets. The raster water mesh
        // does not show those facets because it interpolates the per-vertex
        // cell normals across each Surface-Nets triangle. Reconstruct the same
        // local fan and barycentrically blend its corner normals. Flat water
        // (|n.y| ~ 1) keeps the cheap analytic normal: there are no visible
        // facets to remove and the lake covers most of the screen.
        normal = sdfNormal;
        if (abs(sdfNormal.y) < 0.999) {
            uint rootNode = rmRoots[nearestRoot].node;
            bool nonSurfaceStop;
            uint blendCell = rmFindNodeFromStack(rootNode, nearestNode, hitPos,
                                                 nonSurfaceStop);
            if (blendCell == RM_INVALID_NODE || nonSurfaceStop) blendCell = nearestNode;
            // Displacement/material payload is irrelevant here; only the
            // interpolated normal is consumed.
            RmSurfaceBlend waterBlend = rmBlendAtHit(rootNode, blendCell, hitPos, hitPos);
            if (dot(waterBlend.normal, waterBlend.normal) > 1e-12) {
                normal = normalize(waterBlend.normal);
                if (dot(normal, rd) > 0.0) normal = -normal;
            }
        }
    } else {
        // Material / mapping payload of the hit cell. rmResolveHitShading()
        // reconstructs the same up-to-three-material barycentric blend the
        // raster TCS produces for the triangle under the hit, at the
        // undisplaced base point the raster TES samples.
        int material = max(nearestBrush, 0);
        uint rootNode = rmRoots[nearestRoot].node;
        RmHitShading hit = rmResolveHitShading(rootNode, nearestNode, material,
                                               hitPos, rd, nearestNormal);
        normal = hit.normal;
        rmWriteFragGlobals(hitPos, hit);
    }

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

    // Water layers: the octree stores the water volume, so the marched hit is
    // its top surface. Shade it with the same wind-driven swell + raymarched
    // reflection/refraction response as the raster water surface. The SDF path
    // traces both lobes through the octree itself (no screen-space/previ-
    // ous-frame dependency). RT variants use the exact scene TLAS when ready;
    // otherwise the same SDF raymarch supplies both lobes.
    if (isWater) {
        WaterParamsNamed wp = waterParamsNamed(waterParams[0]);
        vec3 N = normalize(normal);
        vec3 V = -rd;
        // Wind-driven swell: animated interface normal + a small upward-only
        // surface offset (the flat SDF plane carries the same wave motion the
        // raster TES/FS produce from the shared fBm wind field).
        // Screen-space footprint of this pixel on the water plane: the wave
        // field's analytic AA fades any swell band that would alias (the
        // raster path passes dFdx/dFdy; ray hits have no such derivatives, so
        // use the distance and the projection's pixel angle instead).
        float hitDist = length(hitPos - ubo.viewPosition);
        float pixelAngle = 2.0 / max(abs(ubo.viewProjection[1][1]) *
                                     max(pc.viewport.y, 1.0), 1e-6);
        vec2 waveStep = vec2(hitDist * pixelAngle);
        // Same ripple-detail LOD band as the raster surface (WaterSurface.glsl).
        float waveDetail = 1.0 - smoothstep(200.0, 1000.0, hitDist);
        RmWaterSurface wave = rmWaterSurface(wp, hitPos, N, waveDetail,
                                             vec2(waveStep.x, 0.0),
                                             vec2(0.0, waveStep.y));
        vec3 Nw = wave.normal;
        if (!(dot(Nw, Nw) > 0.25)) Nw = N;
        // Never push the mirror origin below the flat interface: a trough
        // below a shallow bed would start the secondary ray inside terrain and
        // produce holes/speckles.
        vec3 shadePos = hitPos + N * min(max(wave.height, 0.0), 1.0);
        vec3 color;
#ifdef RT_ENABLED
        // Use the exact RT resolve whenever the RT scene is ready: it reflects
        // every scene object (not just SDF layers) and resolves the exact bed.
        bool rtResolve = (rt.tlasReady != 0u);
        if (rtResolve) {
            color = rtResolveWaterHit(wp, hitPos, Nw, rd);
        } else
#endif
        {
            color = shadeRaymarchWaterSDF(wp, hitPos, shadePos, N, Nw, V);
        }
        // Sun specular highlight (the raster water adds it separately).
        vec3 toSun = normalize(-ubo.lightDirection);
        vec3 halfDir = normalize(toSun + V);
        float spec = pow(max(dot(Nw, halfDir), 0.0), max(wp.specularPower, 1.0)) *
                     max(wp.specularIntensity, 0.0);
        color += vec3(1.0, 0.97, 0.90) * spec;
        if (any(isnan(color)) || any(isinf(color))) color = wp.waterColor;
        outColor = vec4(clamp(color, 0.0, 1.0), 1.0);
        return;
    }

    outColor = vec4(0.0);
    // Zero override: shadeSolidSurface derives the facet normal from the
    // marched hit positions, exactly like the raster path derives it from the
    // displaced triangle positions (triplanar sign/weights match).
    shadeSolidSurface(vec3(0.0));
}

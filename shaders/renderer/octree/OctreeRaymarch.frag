#version 460

// ── Octree SDF raymarcher ───────────────────────────────────────────────────
// Optional SDF path (Settings: "Octree SDF Raymarch", selected in the Ray
// Marching widget): instead of drawing the tessellated chunks of each layer,
// this fullscreen pass marches the octree's stored corner SDFs for EVERY
// terrain layer (Solid and Water) and shades the nearest isosurface hit. Solid
// layers go through the same shadeSolidSurface() routine the raster path uses
// (materials/mapping, LOD-locked SDF shadows); Water layers use a
// water-surface shading branch (Snell-refracted lake bed + SDF mirror ray,
// composed with the raster surface's Fresnel/strength mix).
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
// Distance LOD: the table stores every Surface level down to the finest leaf
// lattice, but the march only descends while the current cell is larger than
// the screen-space target for the ray distance traveled. Past that point the
// cell's trilinear field is sampled directly, so distant rays skip several
// hierarchy levels instead of walking to lod 0. Near cells still reach the
// leaf lattice and the hit-time ring reconstruction keeps its full detail.
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
// Shared true-displacement field definition: camera and shadow rays evaluate
// the identical displaced field/materials.
#include "../../includes/octree/DisplacementField.glsl"
#include "../../includes/Hsv.glsl"

// Screen-space reflection refinement (set 0, bindings 19/20) required by
// shadeSolidSurface(); reflections fall back to the procedural sky.
layout(set = 0, binding = 19) uniform sampler2D ssrColorTex;
layout(set = 0, binding = 20) uniform sampler2D ssrDepthTex;

// Set=2: octree node table (bindings declared in RaymarchTable.glsl).

layout(push_constant) uniform OctreeRaymarchPush {
    vec4 viewport;     // xy = solid pass target extent in pixels
    uint rootCount;    // number of layer roots in the set-2 root buffer
    uint tableVersion; // low 32 bits of the table revision (future caches)
    uint nodeCount;    // records uploaded for this frame slot
    uint flags;        // reserved
} pc;

// Instrumentation was removed with the profile shader variants; the march
// keeps the call sites as no-ops so the traversal code stays readable.
#define rmCount(idx)

layout(location = FRAG_OUT_COLOR) out vec4 outColor;

// Terrain shadows are ray marched from the same node table. Declared before
// the shared shading body, defined below once the march helpers exist.
#define SOLID_SHADOW_TRACE 1
float solidShadowTrace(vec3 p, vec3 n, vec3 l, out int shadowHint);

// ── Fragment globals consumed by shadeSolidSurface() ───────────────────────
// The raster path fills these from tessellation varyings; a raymarched hit has
// no vertex stage, so the march result fills them instead. Same names/types as
// SolidRenderer.frag's inputs so the shared shading body compiles unchanged.
vec2 fragUV;
vec3 fragNormal;
vec3 fragPosWorld;
ivec3 fragTexIndices;
ivec3 fragTextureTypes;
vec3 fragPosWorldNotDisplaced;
vec3 fragTexWeights;
vec3 fragHSV;
vec3 fragSharpNormal;
vec3 fragTessLevel;
// Displacement SDF ambient occlusion (1 = open). SolidSurface multiplies it
// with the material AO texture; the raster-only path leaves it at 1.
float fragSdfAO = 1.0;
// Conservative |displacement| bound of the shaded hit (world units), used by
// the shadow-ray bias and debug views.
float fragDisplacementBound = 0.0;

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
// Displaced-band rays are bounded more tightly: the field is no longer a
// distance field inside the band, and a grazing ray can otherwise spend the
// whole 1024-step budget per pixel. 512 is enough for the reference scenes at
// 1–4 px steps; unresolved rays discard instead of tripping the GPU watchdog.
const int RM_DISPLACEMENT_STEPS = 512;
// Secondary rays do not need primary-ray precision: the water bed/reflection
// and sun-shadow traces use a smaller budget so a grazing displaced-band ray
// cannot burn the same worst case as the camera per pixel.
const int RM_SECONDARY_STEPS = 512;
const int RM_SHADOW_STEPS = 256;
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
// Safeguarded secant/bisection refinement budget for a bracketed crossing.
// The loop exits on the residual/width test, so the count is a hard safety
// cap rather than a fixed amount of work.
const int RM_REFINE_STEPS = 16;

// Invocation-local debug telemetry (debug views and tests). The march keeps
// these for the nearest accepted hit; they cost one register each and no
// extra work outside the debug views.
int rmDebugIterations = 0;
float rmDebugStep = 0.0;
float rmDebugResidual = 0.0;
float rmDebugDisplacement = 0.0;
float rmDebugBound = 0.0;
int rmDebugRefineIters = 0;
int rmDebugShadowSteps = 0;
float rmDebugShadowT = 0.0;
// Effective LOD (octree cell edge length) of the surface hit currently being
// shaded. Every shadow ray started from that hit inherits this exact value and
// may never choose another one from ray distance, screen error or traversal
// heuristics. Set by the camera march / secondary-hit shading before
// shadeSolidSurface() runs.
float rmInheritedLodSize = 1.0;
// Shadow-ray diagnostics (debug views only; no effect on the visibility term).
int rmShadowDebugMissingLod = 0;
int rmShadowDebugLodMismatch = 0;
float rmShadowDebugPrimaryLod = 0.0;

// World size of one pixel at distance t. viewProjection[1][1] holds
// 1/tan(fovy/2) and pc.viewport is the solid target extent in pixels.
float rmPixelWorldSize(float t) {
    float pixelAngle = 2.0 / max(abs(ubo.viewProjection[1][1]) * max(pc.viewport.y, 1.0), 1e-6);
    return max(t, 0.0) * pixelAngle;
}

// Distance-adaptive hit tolerance. The node epsilon is the base floor; with a
// screen-space tolerance enabled the band widens with the world size of one
// pixel at the sample distance, so distant and grazing rays stop as soon as
// the crossing is sub-pixel instead of over-resolving it.
float rmHitEpsilon(float nodeSize, float t) {
    float eps = max(nodeSize * 2.0e-3, 1.0e-4);
    float px = ubo.raymarchScreenEps;
    if (px <= 0.0) return eps;
    return max(eps, px * rmPixelWorldSize(t));
}

// Distance-adaptive octree LOD: the smallest cell worth descending into for a
// sample at distance t. The table stores the full Surface hierarchy down to
// the finest leaf lattice; walking every level is expensive and unnecessary
// once a cell covers only a few pixels. Past this target the march samples the
// current node's parent-like trilinear field instead of its stored children.
// Eight pixels keeps near cells at leaf detail while cutting the distant node
// count by several hierarchy levels.
const float RM_LOD_CELL_PIXELS = 8.0;
float rmLodCellSize(float t) {
    return max(rmPixelWorldSize(t) * RM_LOD_CELL_PIXELS, 1.0e-4);
}

// Resolve a node's shading material at a given QEF position. Shared by the
// march's cache (position solved from the node) and the hit-time material
// blend (positions solved once for every ring cell). The classifier itself
// lives in RaymarchMaterial.glsl so OctreeDerive.comp resolves the identical
// index into meta.x. The no-fallback variant compiles this classifier out.
#ifndef OCTREE_NO_FALLBACK
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
#endif

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
#ifdef OCTREE_NO_FALLBACK
    // Defined degradation without the solver. The renderer only selects this
    // variant when the derive pass is available, so this path is defensive.
    return rmNodeMin(id) + 0.5 * rmNodeSize(id);
#else
    rmCount(RM_COUNTER_QEF_SOLVES);
    float sdf[8];
    for (int i = 0; i < 8; ++i) sdf[i] = rmNodeSdf(id, i);
    return sdfCellPosition(sdf, rmNodeMin(id), rmNodeSize(id));
#endif
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
#ifdef OCTREE_NO_FALLBACK
    // No classifier in this variant. A sentinel brush without a payload is a
    // precondition violation (the renderer only selects it when derive ran);
    // fall back to material 0 rather than an out-of-range sentinel index.
    return 0;
#else
    return rmResolveMaterialAt(id, rmCellQef(id));
#endif
}

// Ring material lookup. The payload pass resolved the identical LandBrush /
// NormalBrush classifier result into meta.x, so the hit ring must not rerun the
// branchy Perlin classifier per quadrant cell. Cells without a payload (derive
// unavailable) keep the fragment-side classifier, using the ring's already
// solved QEF position instead of solving it again.
int rmRingMaterial(uint id, vec3 qp) {
    if ((uint(rmNodes[id].meta.w) & RM_FLAG_PAYLOAD) != 0u) return rmNodes[id].meta.x;
#ifdef OCTREE_NO_FALLBACK
    int raw = rmNodes[id].meta.z;
    return (raw == LAND_BRUSH_INDEX || raw == NORMAL_BRUSH) ? 0 : raw;
#else
    return rmResolveMaterialAt(id, qp);
#endif
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

// ── True displacement field ("bend the SDF") ───────────────────────────────
// The displaced surface is the zero set of
//
//     F(p) = d_original(p) - disp(p)
//
// where `disp` is a signed world-space offset: positive moves the surface
// along the base outward normal, negative recesses it. The base field is the
// node's trilinear reconstruction of the stored corner SDFs (a conservative
// approximation of the original signed distance), and the offset is sampled
// from the material height map with the same mapping mode / UV scale the
// surface shader uses. DisplacementField.glsl owns the material resolution and
// sampling so camera and shadow rays evaluate the identical expression.
float rmDisplacementN(uint id, int material, vec3 p, vec3 n) {
    if (material < 0) return 0.0;
    RmDisplacement d = rmDisplacementParams(material);
    if (!d.enabled) return 0.0;
    MaterialNamed m = materialNamed(materials[material]);
    int texType = int(rmNodes[id].hsv0.w + 0.5);
    vec2 uvScale = rmNodes[id].hsv1.xy;
    return rmDisplacementValue(d, m, material, texType, uvScale, p, n, 0.0);
}

// Small wrappers keep the march readable; the actual settings/sampling live in
// the shared include.
float rmDisplacement(uint id, int material, vec3 p) {
    if (!rmDisplacementActive(material)) return 0.0;
    return rmDisplacementN(id, material, p, rmNodeNormal(id, p));
}

float rmDisplacement(uint id, int material, vec3 p, vec3 n) {
    if (!rmDisplacementActive(material)) return 0.0;
    return rmDisplacementN(id, material, p, n);
}

// Bumped trilinear field. `base` receives the unbumped value (used for the
// conservative advance outside the displacement bound); `maxDisp` is the
// maximum |offset| this material can produce (0 = no displacement).
//
// `waterRoot` disables the offset entirely: the water layer is a NormalBrush
// volume whose upward faces resolve to material 0, but the water surface only
// receives the Gerstner shading swell, never the solid material height-map
// displacement. Applying it here shifted the flat lake surface by metres.
//
// DEBUG_MODE_ORIGINAL_SDF_SURFACE deliberately bypasses the offset/CSG so the
// debug view shows the base octree isosurface in the same pass.
//
// `csgField` reports that the returned value is a CSG combination of the
// primary and secondary fields rather than a displaced primary surface. The
// march uses it to pick a conservative feature-clamped step.
float rmFieldEx(uint id, vec3 p, int material, bool waterRoot,
                out float base, out float maxDisp, out bool csgField) {
    base = rmTrilinear(id, p);
    maxDisp = 0.0;
    csgField = false;
    if (waterRoot || material < 0 || !rmIsFinite(base)) return base;
    if (ubo.debugMode == DEBUG_MODE_ORIGINAL_SDF_SURFACE) return base;
    RmDisplacement d = rmDisplacementParams(material);
    if (!d.enabled) return base;

    // Secondary SDF CSG: the pattern is a signed field, not a surface offset.
    // Union = min, intersection = max, subtraction = max(primary, -secondary).
    if (!d.legacy && d.pattern > 0 && d.operation != RM_SECONDARY_DISPLACE) {
        csgField = true;
        maxDisp = d.maxAbs; // bound for bias/AO/debug; not a displacement band
        float secondary = rmSecondarySdf(d, p);
        if (!rmIsFinite(secondary)) {
            csgField = false;
            return base;
        }
        if (d.operation == RM_SECONDARY_UNION) return min(base, secondary);
        if (d.operation == RM_SECONDARY_INTERSECTION) return max(base, secondary);
        return max(base, -secondary);
    }

    maxDisp = d.maxAbs;
    // Outside the band |base| > maxDisp, so F = base - disp has the sign of
    // base and |F| >= |base| - maxDisp. Returning the base value is enough for
    // the caller's sign/step logic and skips the height taps entirely on the
    // long outside-band segments. Inside the band the full field is sampled.
    if (maxDisp > 0.0 && abs(base) > maxDisp + max(rmNodeSize(id) * 2.0e-3, 1.0e-4))
        return base;
    MaterialNamed m = materialNamed(materials[material]);
    int texType = int(rmNodes[id].hsv0.w + 0.5);
    vec2 uvScale = rmNodes[id].hsv1.xy;
    float disp = rmDisplacementValue(d, m, material, texType, uvScale, p,
                                     rmNodeNormal(id, p), 0.0);
    return base - disp;
}

float rmField(uint id, vec3 p, int material, bool waterRoot,
              out float base, out float maxDisp) {
    bool csgField;
    return rmFieldEx(id, p, material, waterRoot, base, maxDisp, csgField);
}

// Forward declaration: the legacy displacement wavelength needs the raster
// tessellation level, defined below.
float rmTessLevel(uint id, int material, vec3 p);

// World-space feature wavelength for the conservative in-band step. Uses the
// same settings object as rmField and the legacy tessellation spacing.
float rmDisplacementWavelengthForNode(uint id, int material) {
    RmDisplacement d = rmDisplacementParams(material);
    if (!d.enabled) return 0.0;
    float legacyLevels = 1.0;
    if (d.legacy) {
        legacyLevels = clamp(rmTessLevel(id, material, rmNodeMin(id)) * 16.0, 1.0, 64.0);
    }
    return rmDisplacementWavelength(d, rmNodes[id].hsv1.xy, rmNodeSize(id), legacyLevels);
}

// Gradient of the combined field for a CSG secondary hit. The primary normal
// (trilinear gradient) is not valid once min/max selects the secondary field,
// so use a regular-tetrahedron finite difference of rmFieldEx itself. The
// stencil is clamped to half the owning cell to avoid mixing neighbour cells.
vec3 rmCsgNormal(uint id, vec3 p, int material, float eps) {
    const vec3 kTetra[4] = vec3[4](
        vec3( 0.57735026919,  0.57735026919,  0.57735026919),
        vec3( 0.57735026919, -0.57735026919, -0.57735026919),
        vec3(-0.57735026919,  0.57735026919, -0.57735026919),
        vec3(-0.57735026919, -0.57735026919,  0.57735026919)
    );

    vec3 mn = rmNodeMin(id);
    vec3 cellMax = mn + vec3(rmNodeSize(id));
    vec3 dFace = min(p - mn, cellMax - p);
    float faceDist = max(min(min(dFace.x, dFace.y), dFace.z), 0.0);
    float h = max(min(eps, faceDist * 0.5), 1.0e-6);

    float b0, m0;
    bool c0;
    float f0 = rmFieldEx(id, p, material, false, b0, m0, c0);
    if (!rmIsFinite(f0)) return rmNodeNormal(id, p);

    vec3 grad = vec3(0.0);
    for (int i = 0; i < 4; ++i) {
        float b, m;
        bool c;
        float fi = rmFieldEx(id, p + kTetra[i] * h, material, false, b, m, c);
        if (!rmIsFinite(fi)) continue;
        grad += (fi - f0) * kTetra[i];
    }
    grad *= 3.0 / (4.0 * h);
    float gl = dot(grad, grad);
    if (!(gl > 1.0e-12) || isnan(gl) || isinf(gl)) return rmNodeNormal(id, p);
    return normalize(grad);
}

// Secant refinement of a hit accepted inside the (possibly screen-space
// widened) tolerance band. The march only knows the last two field samples of
// the same node, prevT/dPrev and t/d, which bracket the crossing when the band
// is wider than the step; a short secant iteration recovers the sub-band
// crossing, clamped to the band so the hit can never leave it. Only called
// when the screen-space tolerance is active; the plain node epsilon keeps the
// historical acceptance at the sampled t.
float rmRefineHit(uint node, vec3 ro, vec3 rd, int material, bool waterRoot,
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
        float fn = rmField(node, ro + rd * tn, material, waterRoot, bBase, bMax);
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

// True displaced normal: N = normalize(grad(F)), F = base - disp.
//
// The base trilinear gradient is already available (rmNodeNormal). The
// displacement gradient is estimated with a regular-tetrahedron finite
// difference of the scalar displacement field:
//
//   grad(f) ~= (3 / 4h) * sum_i (f(p + h e_i) - f(p)) e_i
//
// where e_i are the four unit tetra directions (sum e_i e_i^T = 4/3 I). Four
// displacement samples give a stable, orientation-independent gradient without
// an axis-aligned stencil. `eps` is the world sample offset: large enough to
// avoid texture quantization noise, small enough to resolve the material's
// declared feature band (callers pass ~half a pixel at the hit distance).
vec3 rmDisplacedNormal(uint id, int material, vec3 p, vec3 baseNormal,
                       float eps, out float dispAtHit) {
    dispAtHit = 0.0;
    if (material < 0 || !rmDisplacementActive(material)) return baseNormal;
    // CSG secondary hits use the combined-field gradient instead.
    if (rmSecondaryCsgActive(material)) return baseNormal;

    RmDisplacement d = rmDisplacementParams(material);
    if (!d.enabled) return baseNormal;
    MaterialNamed m = materialNamed(materials[material]);
    int texType = int(rmNodes[id].hsv0.w + 0.5);
    vec2 uvScale = rmNodes[id].hsv1.xy;

    const vec3 kTetra[4] = vec3[4](
        vec3( 0.57735026919,  0.57735026919,  0.57735026919),
        vec3( 0.57735026919, -0.57735026919, -0.57735026919),
        vec3(-0.57735026919,  0.57735026919, -0.57735026919),
        vec3(-0.57735026919, -0.57735026919,  0.57735026919)
    );

    // Keep the tetra stencil inside the owning cell when the hit is close to a
    // shared face: the trilinear polynomial is defined outside, but its
    // gradient is not the neighbour's, and mixing cells would create a normal
    // seam exactly at the boundary.
    vec3 mn = rmNodeMin(id);
    vec3 cellMax = mn + vec3(rmNodeSize(id));
    vec3 dFace = min(p - mn, cellMax - p);
    float faceDist = max(min(min(dFace.x, dFace.y), dFace.z), 0.0);
    float h = max(min(eps, faceDist * 0.5), 1e-6);
    float f0 = rmDisplacementValue(d, m, material, texType, uvScale, p,
                                   baseNormal, 0.0);
    dispAtHit = f0;

    vec3 gradDisp = vec3(0.0);
    for (int i = 0; i < 4; ++i) {
        vec3 q = p + kTetra[i] * h;
        // Recompute the base normal at the sample so the triplanar weights
        // follow the real surface orientation instead of freezing the hit
        // basis (the field itself does the same during the march).
        vec3 nq = rmNodeNormal(id, q);
        float fi = rmDisplacementValue(d, m, material, texType, uvScale, q, nq, 0.0);
        gradDisp += (fi - f0) * kTetra[i];
    }
    gradDisp *= 3.0 / (4.0 * h);

    vec3 g = baseNormal - gradDisp;
    float gl = dot(g, g);
    if (!(gl > 1e-12) || isnan(gl) || isinf(gl)) return baseNormal;
    return normalize(g);
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

// `applyDisplacement` mirrors rmField's gate: the water layer's NormalBrush
// cells resolve to material 0, but the raster water surface never receives the
// solid material height-map displacement, so its ring corners must stay on the
// flat interface.
RmSurfaceBlend rmBlendAtHit(uint root, uint hitId, vec3 baseHit, vec3 displacedHit,
                            bool applyDisplacement) {
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
            float disp = applyDisplacement ? rmDisplacementN(id, m, qp, n) : 0.0;
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

// Blue -> green -> red heat ramp for the LOD diagnostics.
vec3 rmLodHeat(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 heat = mix(vec3(0.15, 0.25, 1.0), vec3(0.0, 1.0, 0.25),
                    clamp(t * 2.0, 0.0, 1.0));
    return mix(heat, vec3(1.0, 0.15, 0.0), clamp(t * 2.0 - 1.0, 0.0, 1.0));
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

// Safeguarded root refinement for a sign-change bracket [t0,t1] of the
// displaced field. The first two iterations bisect (guaranteed bracket
// reduction); later iterations try a secant step clamped to the bracket and
// fall back to bisection when the secant leaves it. The loop stops on the
// residual or interval-width tolerance, so a hard iteration cap never
// determines ordinary precision.
float rmRefineCrossing(uint node, vec3 ro, vec3 rd, int material, bool waterRoot,
                       float t0, float f0, float t1, float f1, float eps,
                       out float residual, out int iters) {
    float lo = t0, flo = f0;
    float hi = t1, fhi = f1;
    residual = abs(f1);
    iters = 0;
    float fieldEps = max(eps * 0.05, 1.0e-6);
    for (int i = 0; i < RM_REFINE_STEPS; ++i) {
        if (hi - lo <= fieldEps) break;
        float t;
        float den = fhi - flo;
        if (i >= 2 && abs(den) > 1.0e-12) {
            t = hi - fhi * (hi - lo) / den;
            if (!(t > lo && t < hi)) t = 0.5 * (lo + hi);
        } else {
            t = 0.5 * (lo + hi);
        }
        rmCount(RM_COUNTER_BISECT_EVALS);
        float bBase, bMax;
        float fm = rmField(node, ro + rd * t, material, waterRoot, bBase, bMax);
        ++iters;
        if (isnan(fm)) break;
        if (fm > 0.0) { lo = t; flo = fm; }
        else { hi = t; fhi = fm; }
        residual = abs(fm);
        if (residual <= fieldEps) break;
    }
    float tHit = 0.5 * (lo + hi);
    // Final residual at the returned parameter (debug/validation). The
    // caller's hit position uses this midpoint, not the last secant sample.
    float bBase, bMax;
    float fr = rmField(node, ro + rd * tHit, material, waterRoot, bBase, bMax);
    if (!isnan(fr)) residual = abs(fr);
    return tHit;
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
// Hit acceptance with the actual displaced normal. `baseNormal` is the
// trilinear cell gradient (used to recover the undisplaced shading point);
// `normal` is normal(grad(base - displacement)) and is what visibility,
// lighting, AO, shadow bias and secondary rays consume.
bool rmHitAllowed(uint node, vec3 p, vec3 rd, int material, bool waterRoot,
                  float normalEps, out vec3 baseNormal, out vec3 normal,
                  out float displacement) {
    baseNormal = rmNodeNormal(node, p);
    normal = baseNormal;
    displacement = 0.0;
    if (waterRoot) return abs(normalize(baseNormal).y) > 1.0e-3;
    if (material == RAYMARCH_DISCARD_BRUSH) return false;
    if (ubo.debugMode == DEBUG_MODE_ORIGINAL_SDF_SURFACE) {
        // The field bypasses displacement/CSG in this view; keep the
        // visibility test on the base normal so it shows the original
        // isosurface.
        return dot(normalize(baseNormal), rd) < 0.0;
    }
    if (rmSecondaryCsgActive(material)) {
        // The combined field's gradient is the true surface normal.
        normal = rmCsgNormal(node, p, material, normalEps);
        return dot(normalize(normal), rd) < 0.0;
    }
    normal = rmDisplacedNormal(node, material, p, baseNormal, normalEps,
                               displacement);
    return dot(normalize(normal), rd) < 0.0;
}

bool rmMarchRoot(uint root, vec3 ro, vec3 rd, vec3 invRd,
                 float tMax, bool haveRootInterval, float rootT0, float rootT1,
                 out uint hitNode, out float hitT, out int hitBrush,
                 out vec3 hitNormal, bool waterRoot, int stepBudget) {
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
    bool cachedCsg = false;

    // Previous accepted sample of the CURRENT node, for the secant refinement
    // of a hit accepted inside a screen-space-widened tolerance band. Reset on
    // every discard path so a stale bracket is never reused.
    uint prevNode = 0xFFFFFFFFu;
    float prevT = -1.0;
    float prevD = 0.0;

    for (int iter = 0; iter < stepBudget; ++iter) {
        ++rmDebugShadowSteps; // reset by solidShadowTrace; debug telemetry only
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
        // step that a coarse parent field (pruned children or the distance LOD
        // gate) produces. Clamp every step to this child's exit so the ray
        // visits the next cell (and descends into a Surface child) instead of
        // skipping it. When the LOD gate stops descending, cellExit stays at
        // nodeExit and the parent field is sampled for the whole node.
        float cellExit = nodeExit;

        // Distance-adaptive LOD gate. Once this cell is no larger than the
        // screen-space target for the current ray distance, keep sampling its
        // trilinear field for the rest of the cell instead of descending into
        // the stored children. The children remain available to the hit-time
        // ring reconstruction; only the march itself is coarsened. At close
        // range the target is below the leaf size, so lod 0 is still reached.
        bool descend = !rmNodeIsLeaf(node) && nodeSize > rmLodCellSize(t);

        // Descend into the child cell containing p. Select the cell at a small
        // forward nudge: after a cell skip/exit t lands exactly on a shared
        // face, where floor(local*2) rounds back into the cell being left and
        // the child ray-box test can report c1<=0. The default cellExit then
        // stays at the parent's exit, so the same node is re-entered with a
        // far-too-large interval and every sibling in between is skipped (the
        // observed cause of the missing terrain). The nudge is negligible at
        // cell scale but strictly inside the cell the ray is about to enter.
        if (descend) {
            vec3 mn = rmNodeMin(node);
            float nudge = max(nodeSize * 1.0e-5, 1.0e-4);
            float tn = t + nudge;
            vec3 pn = ro + rd * tn;
            float childSize = nodeSize * 0.5;

            vec3 local = (pn - mn) / max(nodeSize, 1e-8);
            ivec3 cell = clamp(ivec3(floor(local * 2.0)), ivec3(0), ivec3(1));
            int slot = cell.x * 4 + cell.y * 2 + cell.z;
            vec3 childMin = mn + vec3(cell) * childSize;

            if ((pc.flags & 2u) != 0u) {
                // Phase 4 F5: only the child's far-plane exit is needed
                // because pn is already inside the selected cell. Pick the far
                // plane by ray direction instead of running the full slab
                // entry test. childMin and the plane arithmetic match the
                // slab's a/b values, so the exit is bit-identical.
                float c1 = 1.0e30;
                bool valid = true;
                bool haveAxis = false;
                if (abs(rd.x) > 1.0e-9) {
                    float plane = rd.x > 0.0 ? childMin.x + childSize : childMin.x;
                    float d = (plane - pn.x) * invRd.x;
                    if (!(d > 0.0)) valid = false;
                    else { c1 = min(c1, d); haveAxis = true; }
                }
                if (abs(rd.y) > 1.0e-9) {
                    float plane = rd.y > 0.0 ? childMin.y + childSize : childMin.y;
                    float d = (plane - pn.y) * invRd.y;
                    if (!(d > 0.0)) valid = false;
                    else { c1 = min(c1, d); haveAxis = true; }
                }
                if (abs(rd.z) > 1.0e-9) {
                    float plane = rd.z > 0.0 ? childMin.z + childSize : childMin.z;
                    float d = (plane - pn.z) * invRd.z;
                    if (!(d > 0.0)) valid = false;
                    else { c1 = min(c1, d); haveAxis = true; }
                }
                if (!valid || !haveAxis) {
                    // Degenerate ray/plane result: advance a hair and retry.
                    t = min(t + nudge, nodeExit);
                    continue;
                }
                cellExit = min(nodeExit, tn + c1);
            } else {
                // Exit of the child cell measured from the nudged point. It is
                // strictly inside the cell, so the slab test is well defined
                // even when the un-nudged ray sits on a face.
                float c0, c1;
                if (rmRayBoxInv(childMin, childSize, pn, invRd, c0, c1) && c1 > 0.0) {
                    cellExit = min(nodeExit, tn + c1);
                } else {
                    // Degenerate slab result: advance a hair and re-evaluate
                    // the cell next iteration instead of skipping to the
                    // parent exit.
                    t = min(t + nudge, nodeExit);
                    continue;
                }
            }
            int child = rmNodeChild(node, slot);

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
                // Water roots never carry the solid material bump (see
                // rmField), so their Empty/Solid children keep the exact skip
                // and must not fall back to the parent field.
                int bumpBrush = rmCachedBrush(node);
                bool bumpActive = !waterRoot && bumpBrush >= 0 &&
                    rmDisplacementActive(bumpBrush);
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
        bool csgD;
        float d;
        if (node == cachedNode && t == cachedT) {
            // Exact reuse: same node, same parameter, same expression.
            rmCount(RM_COUNTER_CARRY_HITS);
            d = cachedD;
            baseD = cachedBase;
            maxDisp = cachedMaxDisp;
            csgD = cachedCsg;
        } else {
            rmCount(RM_COUNTER_FIELD_EVALS);
            d = rmFieldEx(node, p, material, waterRoot, baseD, maxDisp, csgD);
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
                tHit = rmRefineHit(node, ro, rd, material, waterRoot,
                                   prevT, prevD, t, d, eps);
            // Raster-parity visibility: solid back faces and DISCARD_BRUSH_INDEX
            // cells are not shaded; water accepts only the +Y surface (top
            // plane, from either side), not the volume sides/bottom.
            float normalEps = clamp(rmPixelWorldSize(tHit) * 0.5,
                                    nodeSize * 2.0e-4, nodeSize * 0.05);
            vec3 candidateBase, candidateNormal;
            float candidateDisp;
            if (rmHitAllowed(node, ro + rd * tHit, rd, material, waterRoot,
                             normalEps, candidateBase, candidateNormal,
                             candidateDisp)) {
                hitNode = node;
                hitT = tHit;
                hitBrush = material;
                hitNormal = candidateNormal;
                rmDebugIterations = iter;
                rmDebugStep = (prevNode == node) ? abs(t - prevT) : 0.0;
                rmDebugResidual = abs(d);
                rmDebugRefineIters = 0;
                rmDebugDisplacement = candidateDisp;
                rmDebugBound = maxDisp;
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

        // Conservative sphere trace. Without displacement the trilinear field
        // is (locally) a signed distance and |d| / L cannot cross the surface.
        // With displacement F = base - disp:
        //   * outside the band |base| > maxDisp, |disp| <= maxDisp, so F has
        //     the sign of base and the guaranteed-positive remainder
        //     (|base| - maxDisp) can be advanced safely;
        //   * inside the band the field is not a distance field. We use the
        //     Lipschitz estimate L <= L_base + L_disp with
        //     L_disp ~= 2*pi*maxDisp / wavelength, and clamp the step to a
        //     fraction of the material's declared feature wavelength. The
        //     clamp is the practical band limit: displacement finer than the
        //     declared frequency / pixel footprint is not resolvable and is
        //     documented as filtered detail, not silently missed geometry.
        // Every step is additionally bounded by the child-cell exit below.
        float invLip = ((uint(rmNodes[node].meta.w) & RM_FLAG_PAYLOAD) != 0u)
            ? rmNodes[node].hsv1.z
            : 0.5;
        float stepLen;
        if (csgD) {
            // CSG secondary field: not a displacement band. Use a half-
            // distance step clamped to the pattern's declared feature size;
            // the min/max combination can have a larger gradient than either
            // input near the seam, so half-distance is conservative.
            float wavelength = rmDisplacementWavelengthForNode(node, material);
            if (!(wavelength > 0.0) || isnan(wavelength) || isinf(wavelength))
                wavelength = nodeSize * 0.5;
            float feature = clamp(wavelength, nodeSize * 1.0e-4,
                                  nodeSize * 0.5);
            stepLen = clamp(abs(d) * 0.5, eps, max(feature * 0.5, eps));
        } else if (maxDisp > 0.0) {
            float safe = abs(baseD) - maxDisp;
            if (safe > eps) {
                // Outside the band: base's own bound is exact for F's sign.
                stepLen = max(safe * invLip, eps);
            } else {
                // Inside the band. `wavelength` is the world-space feature
                // size from the material frequency / legacy tess spacing.
                float wavelength = rmDisplacementWavelengthForNode(node, material);
                if (!(wavelength > 0.0) || isnan(wavelength) || isinf(wavelength))
                    wavelength = nodeSize * 0.5;
                // Do not resolve detail below the node's stored scale or the
                // current pixel footprint.
                float feature = clamp(wavelength, nodeSize * 1.0e-4,
                                      nodeSize * 0.5);
                // Never resolve displacement below a few pixels. The step is
                // 1/8..1/2 of `feature`, so flooring at 16 pixels keeps in-band
                // steps at 2..8 pixels instead of dozens of sub-pixel samples
                // (the latter can trip the GPU watchdog on large terrain).
                // Shadow/secondary rays advance t from the hit, so the pixel
                // footprint at the primary hit distance is the meaningful
                // floor there; using t alone would collapse it near the hit.
                float viewDist = length(ro - ubo.viewPosition);
                float pixelFloor = max(rmPixelWorldSize(t),
                                       rmPixelWorldSize(viewDist));
                // Scale patterns are authored macro detail: their gaps are
                // only a fraction of a repeat. Keep a 4-pixel floor for them
                // (the 16-pixel floor is for texture displacement, where fine
                // sub-pixel detail is intentionally filtered).
                float patternFlag = (material >= 0)
                    ? materials[material].patternParams.x : 0.0;
                feature = max(feature, pixelFloor * (patternFlag > 0.5 ? 2.0 : 16.0));
                float Lbase = 1.0 / max(invLip, 1.0e-3);
                float Ldisp = 6.28318530718 * maxDisp / max(feature, 1.0e-4);
                float L = max(Lbase + Ldisp, 1.0e-4);
                // At least 4 samples per feature; never smaller than 1/8 of
                // it (the bracket refinement resolves the exact crossing).
                float stepBand = clamp(abs(d) / L, feature * 0.125,
                                       feature * 0.5);
                stepLen = max(stepBand, eps);
            }
        } else {
            // Payload cache: 1/L for the node's conservative trilinear
            // Lipschitz bound (OctreeDerive.comp), so smooth cells take steps
            // up to 1.155x longer and rough cells step shorter (safer) than
            // the fixed 0.5 factor. Cells without a payload keep 0.5.
            stepLen = max(abs(d) * invLip, eps);
        }
        float tNext = min(t + stepLen, cellExit);
        // Numerical guard: a zero/negative step would spin the bounded loop
        // for its whole budget on one sample. Advance a hair or leave the cell.
        if (!(tNext > t)) {
            float tAdvance = min(t + max(eps, 1.0e-5), nodeExit);
            if (!(tAdvance > t)) return false;
            t = tAdvance;
            prevNode = 0xFFFFFFFFu;
            continue;
        }
        float dNextBase;
        float dNextMax;
        bool dNextCsg;
        rmCount(RM_COUNTER_FIELD_EVALS);
        float dNext = rmFieldEx(node, ro + rd * tNext, material, waterRoot,
                                dNextBase, dNextMax, dNextCsg);
        if (isnan(dNext)) {
            // Cache only finite values: the displacement branch consumes
            // baseD/maxDisp too, and a NaN there would poison the next reuse.
            dNext = -1.0;
            dNextBase = -1.0;
            dNextMax = 0.0;
            dNextCsg = false;
        }

        // Sign change inside this cell: safeguarded secant/bisection refine.
        if (d > 0.0 && dNext <= 0.0) {
            float refineResidual;
            int refineIters;
            float tHit = rmRefineCrossing(node, ro, rd, material, waterRoot,
                                          t, d, tNext, dNext, eps,
                                          refineResidual, refineIters);
            float normalEps = clamp(rmPixelWorldSize(tHit) * 0.5,
                                    nodeSize * 2.0e-4, nodeSize * 0.05);
            vec3 candidateBase, candidateNormal;
            float candidateDisp;
            if (rmHitAllowed(node, ro + rd * tHit, rd, material, waterRoot,
                             normalEps, candidateBase, candidateNormal,
                             candidateDisp)) {
                hitNode = node;
                hitT = tHit;
                hitBrush = material;
                hitNormal = candidateNormal;
                rmDebugIterations = iter;
                rmDebugStep = stepLen;
                rmDebugResidual = refineResidual;
                rmDebugRefineIters = refineIters;
                rmDebugDisplacement = candidateDisp;
                rmDebugBound = maxDisp;
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
        cachedCsg = dNextCsg;
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
    float ao;        // SDF occlusion of the displaced surface (1 = open)
    float dispBound; // |displacement| bound (shadow bias / debug)
};

// SDF ambient occlusion of the displaced surface. Four hemisphere probes on a
// rotated tangent basis sample the SAME field the camera march uses; a probe
// that lands close to or inside the surface contributes occlusion. A plane
// (field ~= distance) contributes nothing because the probe threshold is
// below the geometric distance along a tilted ray, so only real displaced
// overhangs / peaks darken the crevices. This is multiplied with the material
// AO texture; it is not a replacement for it.
float rmSdfAO(uint rootNode, uint hitNode, vec3 p, vec3 n, int material,
              float radius) {
    if (ubo.ambientOcclusionEnabled == 0u || ubo.displacementAOEnabled == 0u ||
        !rmDisplacementActive(material) || !(radius > 0.0)) return 1.0;

    vec3 up = abs(n.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 t = normalize(cross(up, n));
    vec3 b = cross(n, t);

    float occ = 0.0;
    float thresh = radius * 0.35;
    for (int i = 0; i < 4; ++i) {
        float ang = (float(i) + 0.5) * 1.57079632679;
        vec3 dir = normalize(n * 0.8 +
                             (cos(ang) * t + sin(ang) * b) * 0.6);
        vec3 q = p + dir * radius;
        bool nonSurface;
        uint id = rmFindNodeFromStack(rootNode, hitNode, q, nonSurface);
        if (id == RM_INVALID_NODE || nonSurface) continue;
        int probeMat = rmRingMaterial(id, q);
        float bBase, bMax;
        float f = rmField(id, q, probeMat, false, bBase, bMax);
        if (!rmIsFinite(f)) continue;
        float o = clamp((thresh - f) / thresh, 0.0, 1.0);
        occ += o * o;
    }
    return clamp(1.0 - 0.5 * occ, 0.0, 1.0);
}

RmHitShading rmResolveHitShading(uint rootNode, uint node, int material,
                                 vec3 hitPos, vec3 rd, vec3 acceptedNormal) {
    RmHitShading h;
    // Recover the UNDISPLACED base point for material sampling: the height map
    // is evaluated at the base vertex and the offset is applied along the base
    // normal. Two fixed-point steps converge for offsets small relative to the
    // feature scale. `acceptedNormal` is the displaced normal (the true
    // surface normal); the base normal is recomputed from the trilinear field
    // and used ONLY for this inverse mapping, never for lighting.
    const bool csg = rmSecondaryCsgActive(material);
    vec3 baseNormal = rmNodeNormal(node, hitPos);
    if (dot(baseNormal, baseNormal) <= 1e-12) {
        baseNormal = (dot(acceptedNormal, acceptedNormal) > 1e-12)
            ? normalize(acceptedNormal) : vec3(0.0, 1.0, 0.0);
    }
    float hitDisplacement = csg ? 0.0
                                : rmDisplacement(node, material, hitPos, baseNormal);
    h.basePos = hitPos;
    if (hitDisplacement != 0.0) {
        h.basePos = hitPos - baseNormal * hitDisplacement;
        for (int i = 0; i < 2 && hitDisplacement != 0.0; ++i) {
            baseNormal = rmNodeNormal(node, h.basePos);
            hitDisplacement = rmDisplacement(node, material, h.basePos, baseNormal);
            h.basePos = hitPos - baseNormal * hitDisplacement;
        }
    }

    // The raster material blend belongs to the base (undisplaced) cell: use
    // that cell when it is in the table, otherwise the hit's own cell.
    bool baseCellNonSurface;
    uint blendCell = rmFindNodeFromStack(rootNode, node, h.basePos, baseCellNonSurface);
    if (blendCell == RM_INVALID_NODE || baseCellNonSurface) blendCell = node;
    // A CSG hit has no undisplaced base point; the ring must not apply the
    // surface-displacement corner offset.
    h.blend = rmBlendAtHit(rootNode, blendCell, h.basePos, hitPos, !csg);

    // True displacement: use normal(grad(base - disp)) computed by
    // rmDisplacedNormal at the accepted hit. The ring's per-vertex base
    // normals are only a raster-parity fallback for the undisplaced path.
    if (csg) {
        // rmHitAllowed already solved the combined-field gradient.
        h.normal = (dot(acceptedNormal, acceptedNormal) > 1e-12)
            ? normalize(acceptedNormal) : baseNormal;
    } else if (rmDisplacementActive(material)) {
        float hitDist = length(hitPos - ubo.viewPosition);
        float normalEps = clamp(rmPixelWorldSize(hitDist) * 0.5,
                                rmNodeSize(node) * 2.0e-4,
                                rmNodeSize(node) * 0.05);
        float dispAtHit;
        h.normal = rmDisplacedNormal(node, material, hitPos, baseNormal,
                                     normalEps, dispAtHit);
    } else {
        vec3 fallbackNormal = baseNormal;
        if (dot(fallbackNormal, fallbackNormal) <= 1e-12) {
            fallbackNormal = rmNodeNormal(node, hitPos);
        }
        h.normal = (dot(h.blend.normal, h.blend.normal) > 1e-12)
            ? normalize(h.blend.normal) : fallbackNormal;
    }
    if (dot(h.normal, rd) > 0.0) h.normal = -h.normal; // front surface only
    h.blendCell = blendCell;

    // Displacement-aware SDF AO: probe radius covers the material's offset
    // amplitude and half a feature wavelength, clamped to the hit cell so the
    // probes stay in the node table region the march already sampled.
    RmDisplacement d = rmDisplacementParams(material);
    // CSG hits are not a surface offset; the shadow bias must follow the
    // inherited cell size, not the (potentially huge) secondary bounds.
    h.dispBound = csg ? 0.0 : d.maxAbs;
    h.ao = 1.0;
    if (d.enabled) {
        float wavelength = rmDisplacementWavelengthForNode(node, material);
        float radius = max(d.maxAbs * 2.0, wavelength * 0.5);
        radius = clamp(radius, rmNodeSize(node) * 5.0e-4,
                       rmNodeSize(node) * 0.5);
        h.ao = rmSdfAO(rootNode, node, hitPos, h.normal, material, radius);
    }
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
    // Same feed the TES writes (tess levels / 16, clamped to the heat range).
    // The raster's tess levels are per base cell, not per bumped hit.
    fragTessLevel = vec3(rmTessLevel(h.blendCell, rmCachedBrush(h.blendCell), h.basePos));
    fragSdfAO = h.ao;
    fragDisplacementBound = h.dispBound;
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
// isosurface exactly, but solid tessellation displacement folds an offset into
// the field and can move the surface outside those cells. While displacement is
// active the only safe rejection box for a SOLID root is the root cube itself,
// which is the interval the march actually samples anyway. Water roots never
// take the material bump (rmField), so their surface AABB stays exact and also
// keeps the march out of the empty water-volume cells.
// True when rmRootTestBox returns the root cube interval (so the caller's
// slab result can be reused by rmMarchRoot).
bool rmRootBoxIsCube() {
    return ubo.displacementEnabled != 0u || ubo.tessellationEnabled != 0u;
}

void rmRootTestBox(uint root, out vec3 mn, out vec3 mx) {
    // The stored surface AABB bounds the undisplaced iso-surface exactly.
    // Displacement can move it by at most ubo.displacementMaxBound, so expand
    // the AABB by that bound instead of falling back to the whole root cube.
    // The expanded box still bounds every reachable displaced point, but keeps
    // the rejection tight for camera and shadow rays.
    const bool displacedSolid =
        (ubo.displacementEnabled != 0u || ubo.tessellationEnabled != 0u) &&
        rmRoots[root].kind != 1u;
    mn = rmRoots[root].surfaceMin.xyz;
    mx = rmRoots[root].surfaceMax.xyz;
    if (displacedSolid) {
        vec3 b = vec3(max(ubo.displacementMaxBound, 0.0));
        mn -= b;
        mx += b;
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
                        rmRootBoxIsCube(), iv[k].t0, iv[k].t1,
                        node, t, brush, normal, false, RM_SECONDARY_STEPS) && t < hitT) {
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
                        rmRootBoxIsCube(), t0, t1,
                        node, t, brush, normal, false, RM_SECONDARY_STEPS) && t < hitT) {
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

// ── LOD-locked SDF shadow march ───────────────────────────────────────────
// The primary march's effective LOD is the node whose trilinear field produced
// the accepted surface hit (rmInheritedLodSize). A shadow ray must evaluate
// that SAME cell size everywhere along its path:
//
//   * the descent gate is `nodeSize > inheritedCellSize`, with no dependence
//     on ray distance, screen-space error, pixel footprint or camera distance;
//   * once a cell reaches the inherited size its field is sampled directly and
//     the traversal never descends below it (no independent refinement);
//   * a node that is already coarser than the inherited size is never replaced
//     by an available finer child, so the ray cannot switch representations;
//   * when the hierarchy has no stored cell at the inherited size (leaf above
//     it, pruned/non-surface child, cross-layer level mismatch), the region is
//     sampled with the coarsest field that actually exists and
//     rmShadowDebugMissingLod is raised. This is the explicit, visible
//     fallback mandated by the LOD contract — never a silent substitution.
//
// The field expression, material resolution, displacement, corner-sentinel
// handling and conservative stepping are the same helpers the primary march
// uses, so a shadow sample evaluates exactly the same geometric representation.
bool rmShadowMarchRoot(uint root, vec3 ro, vec3 rd, vec3 invRd, float tMax,
                       float lodCell, float biasDist, int stepBudget,
                       out float occlT, inout float visibility) {
    occlT = 0.0;

    // Same conservative root rejection box as the primary shadow path: the
    // stored surface AABB expanded by the global displacement bound.
    vec3 b = vec3(max(ubo.displacementMaxBound, 0.0));
    vec3 boxMin = rmRoots[root].surfaceMin.xyz - b;
    vec3 boxMax = rmRoots[root].surfaceMax.xyz + b;
    float t0, t1;
    if (!rmRayBoxInv(boxMin, boxMax - boxMin, ro, invRd, t0, t1)) return false;
    t1 = min(t1, tMax);
    float t = max(t0, 0.0);
    if (t >= t1) return false;

    uint stackNode[RM_MAX_STACK];
    float stackExit[RM_MAX_STACK];
    int depth = 0;
    uint node = root;
    float nodeExit = t1;

    // Exact cell-size comparison tolerance. The octree subdivides by exact
    // powers of two, so equality is the common case; the tolerance only covers
    // float round-off from the stored cube sizes.
    float lodTol = max(lodCell * 1.0e-3, 1.0e-5);

    for (int iter = 0; iter < RM_SHADOW_STEPS; ++iter) {
        if (iter >= stepBudget) break;
        ++rmDebugShadowSteps;
        if (t >= tMax) return false;
        if (t >= nodeExit) {
            if (depth == 0) return false;
            --depth;
            node = stackNode[depth];
            nodeExit = stackExit[depth];
            continue;
        }

        vec3 p = ro + rd * t;
        float nodeSize = rmNodeSize(node);
        float cellExit = nodeExit;
        const bool coarser = nodeSize > lodCell + lodTol;
        const bool finer = nodeSize < lodCell - lodTol;

        // Fixed-LOD descent: only the inherited cell size decides whether the
        // traversal may enter a child. There is deliberately no `t`-dependent
        // term here.
        bool descend = !rmNodeIsLeaf(node) && coarser;
        if (descend) {
            vec3 mn = rmNodeMin(node);
            float nudge = max(nodeSize * 1.0e-5, 1.0e-4);
            float tn = t + nudge;
            vec3 pn = ro + rd * tn;
            float childSize = nodeSize * 0.5;

            vec3 local = (pn - mn) / max(nodeSize, 1e-8);
            ivec3 cell = clamp(ivec3(floor(local * 2.0)), ivec3(0), ivec3(1));
            int slot = cell.x * 4 + cell.y * 2 + cell.z;
            vec3 childMin = mn + vec3(cell) * childSize;

            float c0, c1;
            if (rmRayBoxInv(childMin, childSize, pn, invRd, c0, c1) && c1 > 0.0) {
                cellExit = min(nodeExit, tn + c1);
            } else {
                t = min(t + nudge, nodeExit);
                continue;
            }

            if (childSize < lodCell - lodTol) {
                // The root's hierarchy cannot represent the inherited cell size
                // (different layer root / mixed level). Do not refine
                // independently; sample this node and flag the missing LOD.
                rmShadowDebugMissingLod = 1;
            } else {
                int child = rmNodeChild(node, slot);
                if (child >= 0) {
                    if (depth < RM_MAX_STACK) {
                        stackNode[depth] = node;
                        stackExit[depth] = nodeExit;
                        ++depth;
                        node = uint(child);
                        nodeExit = cellExit;
                        continue;
                    }
                    // Stack exhausted: keep the parent's field for this child
                    // interval (flagged, as in the primary march).
                    rmShadowDebugMissingLod = 1;
                } else if (rmNodeChildPruned(node, slot)) {
                    // Pruned surface child: the primary march samples the
                    // parent's trilinear field over this interval. Same sample,
                    // same explicit flag.
                    rmShadowDebugMissingLod = 1;
                } else {
                    int childType = rmNodeChildType(node, slot);
                    if (childType != RM_TYPE_SURFACE) {
                        int brush = rmCachedBrush(node);
                        bool displaced = brush >= 0 && rmDisplacementActive(brush);
                        if (!displaced) {
                            if (childType == RM_TYPE_SOLID) {
                                // The unbumped field is entirely inside this
                                // child (the primary march skips it because no
                                // isosurface can cross it). Entering it means the
                                // ray has already crossed the visible surface.
                                occlT = t;
                                return true;
                            }
                            // Empty: no isosurface can cross. Skip exactly.
                            t = max(cellExit, t + nudge);
                            continue;
                        }
                        // Displacement can move the surface across the border,
                        // exactly as in the primary march: keep sampling the
                        // parent field. Coarser than the inherited LOD by one
                        // level; flag it explicitly.
                        rmShadowDebugMissingLod = 1;
                    } else {
                        // Surface child without a stored record (table cap or
                        // missing index): the primary fallback samples the
                        // parent field. Same explicit fallback.
                        rmShadowDebugMissingLod = 1;
                    }
                }
            }
        }

        if (coarser || finer) {
            // Reached a leaf above the inherited LOD, a pruned/non-surface
            // fallback, or a cross-layer level mismatch. The flag makes this
            // visible to the debug views; the sample still uses the only field
            // that exists in the region.
            rmShadowDebugMissingLod = 1;
            rmShadowDebugLodMismatch = 1;
        }

        int material = rmCachedBrush(node);
        float baseD;
        float maxDisp;
        bool csgD;
        float d = rmFieldEx(node, p, material, false, baseD, maxDisp, csgD);
        if (isnan(d)) d = 1.0; // defensive, same policy as the primary march
        // Tolerance follows the cell actually sampled (the inherited LOD in
        // the normal case, the explicit fallback cell in a missing-LOD region).
        float eps = max(nodeSize * 2.0e-3, 1.0e-4);

        if (d <= 0.0) {
            // A lifted origin can still sample marginally inside the surface.
            // Give the bias one interval to clear; only then report occlusion.
            if (t <= biasDist * 1.5) {
                float tAdvance = min(biasDist * 2.0, cellExit);
                if (tAdvance > t) {
                    t = tAdvance;
                    continue;
                }
            }
            occlT = t;
            return true;
        }
        if (abs(d) <= eps) {
            occlT = t;
            return true;
        }

        // Soft-shadow penumbra estimate (IQ-style): the closest approach of the
        // ray to the field bounds the visibility. Skipped when softness is 0
        // (hard shadows) and for the first few bias lengths so contact pixels
        // are not darkened by the lifted origin.
        if (ubo.shadowSoftness > 0.0 && t > max(biasDist * 4.0, eps * 4.0)) {
            visibility = min(visibility,
                             ubo.shadowSoftness * d / max(t, 1.0e-4));
        }

        // Conservative step, identical to the primary march's field rule.
        float invLip = ((uint(rmNodes[node].meta.w) & RM_FLAG_PAYLOAD) != 0u)
            ? rmNodes[node].hsv1.z
            : 0.5;
        float stepLen;
        if (csgD) {
            float wavelength = rmDisplacementWavelengthForNode(node, material);
            if (!(wavelength > 0.0) || isnan(wavelength) || isinf(wavelength))
                wavelength = nodeSize * 0.5;
            float feature = clamp(wavelength, nodeSize * 1.0e-4,
                                  nodeSize * 0.5);
            stepLen = clamp(abs(d) * 0.5, eps, max(feature * 0.5, eps));
        } else if (maxDisp > 0.0) {
            float safe = abs(baseD) - maxDisp;
            if (safe > eps) {
                stepLen = max(safe * invLip, eps);
            } else {
                float wavelength = rmDisplacementWavelengthForNode(node, material);
                if (!(wavelength > 0.0) || isnan(wavelength) || isinf(wavelength))
                    wavelength = nodeSize * 0.5;
                float feature = clamp(wavelength, nodeSize * 1.0e-4,
                                      nodeSize * 0.5);
                float Lbase = 1.0 / max(invLip, 1.0e-3);
                float Ldisp = 6.28318530718 * maxDisp / max(feature, 1.0e-4);
                float L = max(Lbase + Ldisp, 1.0e-4);
                stepLen = max(clamp(abs(d) / L, feature * 0.125,
                                    feature * 0.5), eps);
            }
        } else {
            stepLen = max(abs(d) * invLip, eps);
        }

        float tNext = min(t + stepLen, cellExit);
        if (!(tNext > t)) {
            float tAdvance = min(t + max(eps, 1.0e-5), nodeExit);
            if (!(tAdvance > t)) return false;
            t = tAdvance;
            continue;
        }
        float nb, nm;
        float dNext = rmField(node, ro + rd * tNext, material, false, nb, nm);
        if (isnan(dNext)) dNext = -1.0;

        // Any sign change or inside sample occludes. The shadow does not need
        // the primary's exact crossing position, only visibility.
        if (dNext <= 0.0) {
            if (tNext <= biasDist * 1.5) {
                t = tNext;
                continue;
            }
            occlT = tNext;
            return true;
        }
        t = tNext;
    }
    return false;
}

// Primary-hit sun shadow: march the SOLID layer roots from the shaded point
// toward the light using the inherited primary-hit LOD. Returns 1.0 when the
// light is blocked, 0.0 when it is visible (the convention used by
// shadeSolidSurface). No raster CSM term is sampled: terrain shadows are
// evaluated against the same octree representation that rendered the surface.
float solidShadowTrace(vec3 p, vec3 n, vec3 l, out int shadowHint) {
    shadowHint = 0;
    if (ubo.shadowsEnabled == 0u) return 0.0;

    vec3 toSun = normalize(l);
    float ndl = max(dot(n, toSun), 0.0);
    // A surface facing away from the sun cannot receive direct light; skip the
    // shadow ray entirely (same early-out as the shared shading path).
    if (ndl <= 0.01) return 1.0;

    // Inherited LOD: exactly the cell size of the node that produced the hit.
    // rmInheritedLodSize is written before shading (camera march, water bed or
    // reflection ray). No distance/screen-space term is consulted here.
    const float lodCell = max(rmInheritedLodSize, 1.0e-4);
    rmShadowDebugPrimaryLod = lodCell;
    rmShadowDebugMissingLod = 0;
    rmShadowDebugLodMismatch = 0;
    rmDebugShadowSteps = 0;
    rmDebugShadowT = 0.0;

    // Bias in world units: a fraction of the inherited cell size (local SDF
    // sampling resolution) plus the material's displacement amplitude. This
    // scales with the actual representation instead of a fixed world offset.
    float eps = max(lodCell * 2.0e-3, 1.0e-4);
    float bias = max(eps, fragDisplacementBound * 0.05);
    vec3 ro = p + n * bias + toSun * (bias * 0.05);
    vec3 rd = toSun;
    vec3 invRd = vec3(rmInvComp(rd.x), rmInvComp(rd.y), rmInvComp(rd.z));

    float maxT = (ubo.shadowMaxDistance > 0.0)
        ? min(ubo.shadowMaxDistance, max(ubo.farPlane, 1.0))
        : max(ubo.farPlane, 1.0);
    const int stepBudget = clamp(int(ubo.shadowMaxSteps), 1, RM_SHADOW_STEPS);

    // Front-to-back root intervals among SOLID roots only: water is
    // transparent and must not cast an opaque shadow.
    RmRootInterval iv[RM_MAX_ROOTS];
    int ivCount = 0;
    const uint sortedRoots = min(pc.rootCount, uint(RM_MAX_ROOTS));
    for (uint i = 0u; i < sortedRoots; ++i) {
        if (rmRoots[i].kind != 0u) continue;
        vec3 testMin, testMax;
        rmRootTestBox(i, testMin, testMax);
        float t0, t1;
        if (!rmRayBoxInv(testMin, testMax - testMin, ro, invRd, t0, t1)) continue;
        if (t1 <= 0.0 || t0 >= maxT) continue;
        rmInsertRootInterval(iv, ivCount, i, t0, t1);
    }

    float visibility = 1.0;
    for (int k = 0; k < ivCount; ++k) {
        if (iv[k].t0 >= maxT) break;
        float occlT;
        if (rmShadowMarchRoot(iv[k].root, ro, rd, invRd, maxT, lodCell,
                              bias, stepBudget, occlT, visibility)) {
            rmDebugShadowT = occlT;
            return 1.0;
        }
    }
    // Roots beyond the sorting budget: unsorted fallback, same LOD lock.
    for (uint i = sortedRoots; i < pc.rootCount; ++i) {
        if (rmRoots[i].kind != 0u) continue;
        vec3 testMin, testMax;
        rmRootTestBox(i, testMin, testMax);
        float t0, t1;
        if (!rmRayBoxInv(testMin, testMax - testMin, ro, invRd, t0, t1)) continue;
        if (t1 <= 0.0 || t0 >= maxT) continue;
        float occlT;
        if (rmShadowMarchRoot(i, ro, rd, invRd, maxT, lodCell, bias,
                              stepBudget, occlT, visibility)) {
            rmDebugShadowT = occlT;
            return 1.0;
        }
    }
    return clamp(1.0 - visibility, 0.0, 1.0);
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
    // The effective LOD of this surface hit is the cell that produced it. Any
    // shadow ray started by this shading pass inherits that exact cell size.
    rmInheritedLodSize = rmNodeSize(node);
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
//     mirror mix, and the SDF surface shadow darkens the result.
vec3 shadeRaymarchWaterSDF(WaterParamsNamed wp, vec3 hitPos, vec3 shadePos,
                           vec3 N, vec3 Nw, vec3 V) {
    vec3 toSun = normalize(-ubo.lightDirection);
    vec3 rd = -V;
    float ndv = clamp(dot(Nw, V), 0.0, 1.0);
    // Raster water Fresnel: Schlick base + refraction/reflection mix.
    float fresnel = clamp(0.02 + 0.98 * pow(1.0 - ndv,
                          clamp(wp.fresnelPower, 1.0, 8.0)), 0.0, 1.0);

    // SDF sun shadow on the water surface (the refracted bed carries its own).
    // The ray inherits the water hit's effective LOD; no CSM/map is sampled.
    float surfaceShadow = 0.0;
    if (ubo.shadowsEnabled != 0u) {
        int shadowHint;
        surfaceShadow = solidShadowTrace(hitPos, Nw, toSun, shadowHint);
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
        // triplanar mapping, displacement parity, SDF shadows). It writes
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
        vec3 deep = proceduralSkyGradient(refrDir, sky.horizonColor,
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
            reflCol = proceduralSkyGradient(reflDir, sky.horizonColor,
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
                        rmRootBoxIsCube(), iv[k].t0, iv[k].t1,
                        node, t, brush, normal, rmRoots[i].kind == 1u,
                        (ubo.displacementEnabled != 0u && rmRoots[i].kind != 1u)
                            ? RM_DISPLACEMENT_STEPS : RM_MAX_STEPS) &&
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
                        rmRootBoxIsCube(), t0, t1,
                        node, t, brush, normal, rmRoots[i].kind == 1u,
                        (ubo.displacementEnabled != 0u && rmRoots[i].kind != 1u)
                            ? RM_DISPLACEMENT_STEPS : RM_MAX_STEPS) &&
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
    // Effective primary LOD at the accepted hit. Preserved verbatim so every
    // shadow ray launched while shading this surface evaluates the same octree
    // representation (rmShadowMarchRoot).
    rmInheritedLodSize = rmNodeSize(nearestNode);
    const bool isWater = rmRoots[nearestRoot].kind == 1u;
    // rmHitAllowed() already rejected solid DISCARD_BRUSH_INDEX cells and
    // non-upward water faces. Keep the solid guard as a defensive backstop;
    // water shading is driven by the +Y surface geometry, not the material
    // index (shore cells may still carry the NormalBrush discard value).
    if (!isWater && nearestBrush == RAYMARCH_DISCARD_BRUSH) discard;

    vec3 hitPos = ro + rd * nearestT;

    // Water hits use the analytic SDF normal directly: a water node is not a
    // terrain material cell, so the displacement and brush-blend machinery
    // (which follows a terrain height map) must not perturb its interface
    // normal. Keeping the water normal clean is what makes the
    // reflection/refraction directions stable pixel-to-pixel.
    vec3 normal;
    RmHitShading solidHit;
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
            RmSurfaceBlend waterBlend = rmBlendAtHit(rootNode, blendCell, hitPos,
                                                     hitPos, false);
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
        solidHit = rmResolveHitShading(rootNode, nearestNode, material,
                                       hitPos, rd, nearestNormal);
        normal = solidHit.normal;
        rmWriteFragGlobals(hitPos, solidHit);
    }

    vec4 clip = ubo.viewProjection * vec4(hitPos, 1.0);
    gl_FragDepth = (clip.w > 1e-6) ? clamp(clip.z / clip.w, 0.0, 1.0) : 1.0;

    // ── True-displacement diagnostics ─────────────────────────────────────
    // These views are produced by the same march that shaded the hit, so they
    // show the actual traversal values (not a parallel approximation).
    if (!isWater && ubo.debugMode >= DEBUG_MODE_ORIGINAL_SDF_SURFACE &&
        ubo.debugMode <= DEBUG_MODE_SHADOW_LOD_MISSING) {
        float bound = max(solidHit.dispBound, 1.0e-6);
        int mode = ubo.debugMode;
        if (mode == DEBUG_MODE_ORIGINAL_SDF_SURFACE) {
            vec3 baseN = rmNodeNormal(nearestNode, hitPos);
            outColor = vec4(normalize(baseN) * 0.5 + 0.5, 1.0);
            return;
        }
        if (mode == DEBUG_MODE_DISPLACEMENT_MAGNITUDE) {
            float s = clamp(rmDebugDisplacement / bound, -1.0, 1.0);
            vec3 c = (s >= 0.0) ? mix(vec3(0.5), vec3(0.0, 1.0, 0.0), s)
                                : mix(vec3(0.5), vec3(1.0, 0.0, 0.0), -s);
            outColor = vec4(c, 1.0);
            return;
        }
        if (mode == DEBUG_MODE_DISPLACEMENT_BOUNDS) {
            // Green: the hit sits inside the conservative |disp| band.
            // Red: outside (the base distance property governs the step).
            float inside = step(abs(rmDebugDisplacement), bound + 1.0e-4);
            outColor = vec4(1.0 - inside, inside, 0.0, 1.0);
            return;
        }
        if (mode == DEBUG_MODE_DISPLACED_NORMALS) {
            outColor = vec4(normalize(normal) * 0.5 + 0.5, 1.0);
            return;
        }
        if (mode == DEBUG_MODE_DISPLACEMENT_ITERATIONS) {
            float t = clamp(float(rmDebugIterations) / 128.0, 0.0, 1.0);
            vec3 heat = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), clamp(t * 2.0, 0.0, 1.0));
            heat = mix(heat, vec3(1.0, 0.0, 0.0), clamp(t * 2.0 - 1.0, 0.0, 1.0));
            outColor = vec4(heat, 1.0);
            return;
        }
        if (mode == DEBUG_MODE_DISPLACEMENT_STEP) {
            float px = rmDebugStep / max(rmPixelWorldSize(nearestT), 1.0e-6);
            float t = clamp(px / 16.0, 0.0, 1.0);
            vec3 heat = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0), clamp(t * 2.0, 0.0, 1.0));
            heat = mix(heat, vec3(1.0, 0.0, 0.0), clamp(t * 2.0 - 1.0, 0.0, 1.0));
            outColor = vec4(heat, 1.0);
            return;
        }
        if (mode == DEBUG_MODE_DISPLACEMENT_RESIDUAL) {
            float eps = rmHitEpsilon(rmNodeSize(nearestNode), nearestT);
            float t = clamp(rmDebugResidual / max(eps, 1.0e-6), 0.0, 1.0);
            vec3 heat = mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), t);
            outColor = vec4(heat, 1.0);
            return;
        }
        if (mode == DEBUG_MODE_DISPLACEMENT_SHADOW) {
            int hint;
            float shadow = solidShadowTrace(hitPos, normalize(normal),
                                            normalize(-ubo.lightDirection), hint);
            float steps = clamp(float(rmDebugShadowSteps) / 128.0, 0.0, 1.0);
            outColor = vec4(steps, shadow, 0.0, 1.0);
            return;
        }
        if (mode >= DEBUG_MODE_PRIMARY_LOD && mode <= DEBUG_MODE_SHADOW_LOD_MISSING) {
            vec3 toSun = normalize(-ubo.lightDirection);
            int hint;
            // The shadow trace fills the inherited-LOD / missing-LOD telemetry.
            float shadow = solidShadowTrace(hitPos, normalize(normal), toSun, hint);
            float visibility = clamp(1.0 - shadow, 0.0, 1.0);
            float rootSize = max(rmNodeSize(rmRoots[nearestRoot].node), 1.0e-6);
            float primaryLevel = clamp(log2(rootSize /
                                            max(rmShadowDebugPrimaryLod, 1.0e-6)) / 12.0,
                                       0.0, 1.0);
            if (mode == DEBUG_MODE_PRIMARY_LOD) {
                outColor = vec4(rmLodHeat(primaryLevel), 1.0);
                return;
            }
            if (mode == DEBUG_MODE_SHADOW_LOD) {
                // Green = every sampled shadow cell equals the primary hit's
                // effective cell size; red = a different representation was
                // sampled (the explicit missing-LOD fallback).
                float ok = (rmShadowDebugLodMismatch == 0) ? 1.0 : 0.0;
                outColor = vec4(1.0 - ok, ok, 0.0, 1.0);
                return;
            }
            if (mode == DEBUG_MODE_SHADOW_VISIBILITY) {
                outColor = vec4(1.0 - visibility, visibility, 0.0, 1.0);
                return;
            }
            if (mode == DEBUG_MODE_SHADOW_DISTANCE) {
                float maxT = (ubo.shadowMaxDistance > 0.0)
                    ? ubo.shadowMaxDistance : max(ubo.farPlane, 1.0);
                float t = clamp(rmDebugShadowT / max(maxT, 1.0e-4), 0.0, 1.0);
                vec3 heat = mix(vec3(0.0, 1.0, 0.2), vec3(1.0, 0.0, 0.0), t);
                outColor = vec4(heat, 1.0);
                return;
            }
            // DEBUG_MODE_SHADOW_LOD_MISSING: red marks a sample/region where the
            // inherited cell size was not representable and the explicit
            // coarser fallback ran.
            float missing = (rmShadowDebugMissingLod != 0) ? 1.0 : 0.0;
            outColor = vec4(missing, 1.0 - missing, 0.0, 1.0);
            return;
        }
    }

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
    // ous-frame dependency).
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
        vec3 color = shadeRaymarchWaterSDF(wp, hitPos, shadePos, N, Nw, V);
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

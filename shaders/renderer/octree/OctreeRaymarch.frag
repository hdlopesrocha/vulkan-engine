#version 460

// ── Octree SDF raymarcher ───────────────────────────────────────────────────
// Optional solid path (Settings: "Octree SDF Raymarch"): instead of drawing the
// tessellated solid chunks, this fullscreen pass marches the octree's stored
// corner SDFs and shades the first isosurface hit with the same
// shadeSolidSurface() routine the raster path uses.
//
// The node table (utils/OctreeRaymarchTable.hpp) stores SURFACE cells only,
// flattened breadth-first per enabled Solid layer. Each layer contributes one
// root entry covering a cube; roots are marched independently and the nearest
// hit wins. Non-surface children are not stored: their SpaceType is packed in
// the parent so Empty children (all corners outside) and Solid children (all
// corners inside) are both skipped — by construction the isosurface cannot
// cross either. Only node-cap-pruned Surface children fall back to the
// parent's trilinear field; that fallback is valid because the surface really
// does pass through them, whereas sampling a parent field inside a Solid cell
// invents a crossing (a coarse trilinear patch through solid rock).
//
// Surface hits are depth-tested and write gl_FragDepth, so the water pass, the
// composite and the droplet pass keep their existing solid-depth dependency.
// Misses discard (no color, no depth).

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
#include "../../includes/shadow/Shadows.glsl"
#include "../../includes/Hsv.glsl"

// RT is never enabled for this pass: the raymarch path deliberately skips the
// hardware-RT reflection branch (the octree SDF is the geometry itself, so
// secondary visibility would need a TLAS rebuild per edit). RtParams provides
// the struct/helpers only; the RT uniforms below are compiled out.
#include "../../includes/rt/RtParams.glsl"

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
const int RM_MAX_STEPS = 320;
const int RM_MAX_STACK = 40;
const int RM_BISECT_STEPS = 7;

float rmHitEpsilon(float nodeSize) {
    return max(nodeSize * 2.0e-3, 1.0e-4);
}

// Final material of a node, bit-for-bit the OctreeTriangles.comp
// nodeResolvedBrush() rule: resolve LAND/NORMAL sentinels from the GPU QEF
// position + gradient (so terrain classification matches the raster mesh),
// keep the stored index otherwise. DISCARD_BRUSH means "no surface here".
int rmResolvedBrush(uint id) {
    int raw = rmNodes[id].meta.z;
    if (raw != LAND_BRUSH_INDEX && raw != NORMAL_BRUSH) return raw;

    float sdf[8];
    for (int i = 0; i < 8; ++i) sdf[i] = rmNodeSdf(id, i);
    vec3 mn = rmNodeMin(id);
    float sz = rmNodeSize(id);
    vec3 p = sdfCellPosition(sdf, mn, sz);
    vec3 n = sdfCellNormal(sdf, mn, sz, p);

    if (raw == LAND_BRUSH_INDEX) return landMaterialIndex(p, n);
    // NormalBrush keeps only +Y-facing cells, like the old CPU painter.
    return (dot(normalize(n), vec3(0.0, 1.0, 0.0)) > 0.0) ? 0 : RAYMARCH_DISCARD_BRUSH;
}

vec3 rmNodeNormal(uint id, vec3 p) {
    float sdf[8];
    for (int i = 0; i < 8; ++i) sdf[i] = rmNodeSdf(id, i);
    return sdfCellNormal(sdf, rmNodeMin(id), rmNodeSize(id), p);
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
                // Empty (all-outside) or Solid (all-inside): the isosurface
                // cannot pass through, so skip to the child's ray-box exit
                // without sampling. Pruned SURFACE children also have no
                // stored index, but their pruned bit keeps them out of this
                // branch: they fall through to the parent field below.
                t = max(cellExit, t + nudge); // nudge guards a zero-length skip
                continue;
            }
            // Pruned surface child (also: stack exhausted): keep sampling the
            // parent field, but only up to the end of this child cell.
        }

        float d = rmTrilinear(node, p);
        if (isnan(d)) d = 1.0; // defensive: never let a bad sample freeze the march
        float eps = rmHitEpsilon(nodeSize);

        if (abs(d) <= eps) {
            int brush = rmResolvedBrush(node);
            if (brush != RAYMARCH_DISCARD_BRUSH) {
                hitNode = node;
                hitT = t;
                hitBrush = brush;
                return true;
            }
            t += eps;
            continue;
        }

        // Sphere trace with a safety factor; the field is trilinear, not an
        // exact metric distance, so a full-step march can overshoot. Stop at
        // the child-cell exit so the next iteration can descend into it.
        float stepLen = max(abs(d) * 0.5, eps);
        float tNext = min(t + stepLen, cellExit);
        float dNext = rmTrilinear(node, ro + rd * tNext);
        if (isnan(dNext)) dNext = -1.0;

        // Sign change inside this cell: refine with bisection.
        if (d > 0.0 && dNext <= 0.0) {
            float lo = t;
            float hi = tNext;
            for (int b = 0; b < RM_BISECT_STEPS; ++b) {
                float mid = 0.5 * (lo + hi);
                float dm = rmTrilinear(node, ro + rd * mid);
                if (isnan(dm)) break;
                if (dm > 0.0) lo = mid; else hi = mid;
            }
            float tHit = 0.5 * (lo + hi);
            int brush = rmResolvedBrush(node);
            if (brush != RAYMARCH_DISCARD_BRUSH) {
                hitNode = node;
                hitT = tHit;
                hitBrush = brush;
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
    int nearestBrush = RAYMARCH_DISCARD_BRUSH;
    bool found = false;
    for (uint i = 0u; i < pc.rootCount; ++i) {
        uint node;
        float t;
        int brush;
        if (rmMarchRoot(rmRoots[i], ro, rd, node, t, brush) && t < nearestT) {
            nearestT = t;
            nearestNode = node;
            nearestBrush = brush;
            found = true;
        }
    }
    if (!found || nearestBrush == RAYMARCH_DISCARD_BRUSH) discard;

    vec3 hitPos = ro + rd * nearestT;
    vec3 normal = rmNodeNormal(nearestNode, hitPos);
    if (dot(normal, rd) > 0.0) normal = -normal; // front surface only

    // Material / mapping payload of the hit cell (mirrors the per-vertex
    // values the tessellator emits for the same node). The brush index is a
    // validated material index (the same one the raster vertex would carry),
    // so it is used directly; brushLayer() clamps the texture-array tap.
    int material = max(nearestBrush, 0);
    int textureType = int(rmNodes[nearestNode].hsv0.w + 0.5);

    fragPosWorld = hitPos;
    fragPosWorldNotDisplaced = hitPos;
    fragNormal = normal;
    fragSharpNormal = normal;
    fragTexIndices = ivec3(material);
    fragTextureTypes = ivec3(textureType);
    fragTexWeights = vec3(1.0, 0.0, 0.0);
    fragHSV = rmNodes[nearestNode].hsv0.xyz;
    // Vertex UV scale (OctreeNode::data) used by custom/projection mapping.
    fragUV = rmNodes[nearestNode].hsv1.xy;
    fragPosLightSpace = ubo.lightSpaceMatrix * vec4(hitPos, 1.0);
    fragTessLevel = vec3(0.0);

    vec4 clip = ubo.viewProjection * vec4(hitPos, 1.0);
    gl_FragDepth = (clip.w > 1e-6) ? clamp(clip.z / clip.w, 0.0, 1.0) : 1.0;

    outColor = vec4(0.0);
    shadeSolidSurface(normal);
}

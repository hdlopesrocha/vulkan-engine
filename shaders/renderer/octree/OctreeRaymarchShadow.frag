#version 460

// ── Raymarched terrain caster for the CSM cascades ─────────────────────────
// The raster CSM has no terrain chunks (terrain is SDF-raymarched), so without
// this pass displaced terrain would not cast shadows onto vegetation, SDF
// objects or video layers. For every shadow-map texel this pass reconstructs
// the orthographic light ray from the cascade's inverse light-space matrix and
// intersects the SAME field the camera march uses:
//
//     F(p) = d_original(p) - displacement(p)
//
// The field definition and material resolution are shared through
// includes/octree/DisplacementField.glsl; the node table is the same set=2
// stream as the camera pass. Only the accepted hit's depth is written (EVSM
// moments + gl_FragDepth), so the rasterized receiver keeps its existing CSM
// sampling path unchanged.
//
// No derive payload is required: the caster reads the raw brush index and
// resolves LAND/NORMAL sentinels itself, and it only needs the node's corner
// Lipschitz bound. It can therefore run before OctreeDerive.comp in the frame.

#include "../../includes/octree/RaymarchTable.glsl"
#include "../../includes/SceneBindings.glsl"
#include "../../includes/Textures.glsl"
#include "../../includes/Common.glsl"
#include "../../includes/octree/SdfCell.glsl"
#include "../../includes/octree/LandBrush.glsl"
#include "../../includes/octree/RaymarchMaterial.glsl"
#include "../../includes/octree/DisplacementField.glsl"
#include "../../includes/shadow/EvsmWrite.glsl"

layout(push_constant) uniform OctreeRaymarchPush {
    vec4 viewport;      // xy = shadow-map extent in pixels
    uint rootCount;
    uint tableVersion;
    uint nodeCount;
    uint flags;
    mat4 shadowInvLightSpace; // inverse of this cascade's light-space matrix
} pc;

const int SHADOW_MAX_STEPS = 128;
// The caster only needs macro terrain shadows: a cell/step may cover several
// shadow texels without changing the shadow silhouette, and this keeps the
// fullscreen per-cascade traversal inside the iGPU watchdog budget.
const float SHADOW_LOD_TEXELS = 16.0;
const float SHADOW_STEP_TEXELS = 16.0;
const int SHADOW_MAX_STACK = 16;
const int SHADOW_REFINE_STEPS = 8;

layout(location = 0) out vec2 outEVSM;

// Resolve a node material without the derive payload. The payload (meta.x,
// hsv1.z) is produced once per revision by OctreeDerive.comp, but the shadow
// caster may run before that pass on a fresh revision; only the LAND/NORMAL
// sentinels need the classifier then.
uint shadowBrushNode = 0xFFFFFFFFu;
int shadowBrushValue = -1;

int rmShadowMaterial(uint id) {
    if (id == shadowBrushNode) return shadowBrushValue;
    int mat;
    if ((uint(rmNodes[id].meta.w) & RM_FLAG_PAYLOAD) != 0u) {
        mat = rmNodes[id].meta.x;
    } else {
        int raw = rmNodes[id].meta.z;
        if (raw != LAND_BRUSH_INDEX && raw != NORMAL_BRUSH) {
            mat = raw;
        } else {
            float sdf[8];
            for (int i = 0; i < 8; ++i) sdf[i] = rmNodeSdf(id, i);
            vec3 mn = rmNodeMin(id);
            float sz = rmNodeSize(id);
            vec3 qef = sdfCellPosition(sdf, mn, sz);
            vec3 n = sdfCellNormal(sdf, mn, sz, qef);
            mat = rmResolveMaterialAtCorners(raw, sdf, mn, sz, qef, n);
        }
    }
    shadowBrushNode = id;
    shadowBrushValue = mat;
    return mat;
}

// Trilinear node gradient (the same sdfCellNormal the camera march uses).
vec3 rmShadowNormal(uint id, vec3 p) {
    float sdf[8];
    for (int i = 0; i < 8; ++i) sdf[i] = rmNodeSdf(id, i);
    return sdfCellNormal(sdf, rmNodeMin(id), rmNodeSize(id), p);
}

// Field evaluation, identical to the camera march's rmField (minus the
// per-hit debug branches). `base` is the undisplaced trilinear value.
float rmShadowField(uint id, vec3 p, int material, out float base,
                    out float maxDisp) {
    base = rmTrilinear(id, p);
    maxDisp = 0.0;
    if (material < 0 || !rmIsFinite(base)) return base;
    RmDisplacement d = rmDisplacementParams(material);
    if (!d.enabled) return base;
    maxDisp = d.maxAbs;
    // Same outside-band shortcut as the camera march: the base sign governs
    // F there and no height tap is needed.
    if (maxDisp > 0.0 && abs(base) > maxDisp + max(rmNodeSize(id) * 2.0e-3, 1.0e-4))
        return base;
    MaterialNamed m = materialNamed(materials[material]);
    int texType = int(rmNodes[id].hsv0.w + 0.5);
    vec2 uvScale = rmNodes[id].hsv1.xy;
    float disp = rmDisplacementValue(d, m, material, texType, uvScale, p,
                                     rmShadowNormal(id, p), 0.0);
    return base - disp;
}

// Legacy displacement wavelength uses the same camera-distance tess level the
// camera march uses; ubo.viewPosition and the tess distance range are copied
// into the shadow UBO. New-path frequency is distance-independent.
float rmShadowTessLevel(uint id, int material, vec3 p) {
    MaterialNamed m = materialNamed(materials[material]);
    if (!m.mappingEnabled) return 1.0 / 16.0;
    float d = length(p - ubo.viewPosition);
    float t = clamp(1.0 - smoothstep(ubo.tessNearDist, ubo.tessFarDist, d), 0.0, 1.0);
    float minLevel = m.minLevel * ubo.tessellationFactor;
    float maxLevel = m.maxLevel * ubo.tessellationFactor;
    return clamp((mix(minLevel, maxLevel, t) + m.tessLevel * t) / 16.0, 0.0, 4.0);
}

float rmShadowWavelength(uint id, int material, vec3 p) {
    RmDisplacement d = rmDisplacementParams(material);
    if (!d.enabled) return 0.0;
    float legacyLevels = 1.0;
    if (d.legacy)
        legacyLevels = clamp(rmShadowTessLevel(id, material, p) * 16.0, 1.0, 64.0);
    return rmDisplacementWavelength(d, rmNodes[id].hsv1.xy, rmNodeSize(id),
                                    legacyLevels);
}

// March one solid root. This is the camera march's adaptive descent without
// the hit-shading/ring/water branches: descend into stored Surface children,
// skip Empty/Solid children when the unbumped field cannot cross, sample the
// trilinear+displacement field, take conservative steps and refine the first
// positive -> non-positive crossing.
bool rmShadowMarchRoot(uint root, vec3 ro, vec3 rd, vec3 invRd, float tMax,
                       float texelWorld, out float hitT) {
    // Tight conservative box: the stored surface AABB expanded by the largest
    // material offset. Using the root cube here made every shadow texel whose
    // light ray crossed the 15 km terrain cube march the hierarchy, which
    // dominated the caster and could trip the GPU watchdog on integrated
    // hardware.
    vec3 b = vec3(max(ubo.displacementMaxBound, 0.0));
    vec3 boxMin = rmRoots[root].surfaceMin.xyz - b;
    vec3 boxMax = rmRoots[root].surfaceMax.xyz + b;
    float t0, t1;
    if (!rmRayBoxInv(boxMin, boxMax - boxMin, ro, invRd, t0, t1))
        return false;
    t1 = min(t1, tMax);
    float t = max(t0, 0.0);
    if (t >= t1) return false;

    uint stackNode[SHADOW_MAX_STACK];
    float stackExit[SHADOW_MAX_STACK];
    int depth = 0;
    uint node = root;
    float nodeExit = t1;

    for (int iter = 0; iter < SHADOW_MAX_STEPS; ++iter) {
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

        // Shadow LOD: the cascade texel footprint is large, so stop descending
        // once a cell covers only a few texels. This bounds the caster's
        // per-texel node count without changing the field definition.
        bool descend = !rmNodeIsLeaf(node) &&
                       nodeSize > max(texelWorld * SHADOW_LOD_TEXELS, 1.0e-4);
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
            if (rmRayBoxInv(childMin, childSize, pn, invRd, c0, c1) && c1 > 0.0)
                cellExit = min(nodeExit, tn + c1);
            else {
                t = min(t + nudge, nodeExit);
                continue;
            }

            int child = rmNodeChild(node, slot);
            if (child >= 0) {
                if (depth < SHADOW_MAX_STACK) {
                    stackNode[depth] = node;
                    stackExit[depth] = nodeExit;
                    ++depth;
                    node = uint(child);
                    nodeExit = cellExit;
                    continue;
                }
            } else if (!rmNodeChildPruned(node, slot) &&
                       rmNodeChildType(node, slot) != RM_TYPE_SURFACE) {
                // Empty/Solid child: the unbumped isosurface cannot cross.
                // Displacement can push the surface across the border, so keep
                // sampling the parent field when the node material displaces.
                int brush = rmShadowMaterial(node);
                bool displaced = brush >= 0 && rmDisplacementActive(brush);
                if (!displaced) {
                    t = max(cellExit, t + nudge);
                    continue;
                }
            }
        }

        int material = rmShadowMaterial(node);
        float baseD, maxDisp;
        float d = rmShadowField(node, p, material, baseD, maxDisp);
        if (isnan(d)) d = 1.0;
        float eps = max(nodeSize * 2.0e-3, 1.0e-4);

        if (d <= 0.0) {
            // The ray entered the solid without a bracketed crossing (possible
            // when the origin is inside a displaced overhang). The first
            // non-positive sample is the visible surface for the shadow map.
            hitT = t;
            return true;
        }
        if (abs(d) <= eps) {
            hitT = t;
            return true;
        }

        float invLip = ((uint(rmNodes[node].meta.w) & RM_FLAG_PAYLOAD) != 0u)
            ? rmNodes[node].hsv1.z : 0.5;
        float stepLen;
        if (maxDisp > 0.0) {
            float safe = abs(baseD) - maxDisp;
            if (safe > eps) {
                stepLen = max(safe * invLip, eps);
            } else {
                float wavelength = rmShadowWavelength(node, material, p);
                if (!(wavelength > 0.0) || isnan(wavelength) || isinf(wavelength))
                    wavelength = nodeSize * 0.5;
                float feature = clamp(wavelength, nodeSize * 1.0e-4,
                                      nodeSize * 0.5);
                // Shadow detail below several texels cannot be represented in
                // the cascade; do not pay for it in the caster.
                feature = max(feature, texelWorld * SHADOW_STEP_TEXELS);
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
        float dNext = rmShadowField(node, ro + rd * tNext, material, nb, nm);
        if (isnan(dNext)) dNext = -1.0;

        if (d > 0.0 && dNext <= 0.0) {
            float lo = t, hi = tNext;
            for (int b = 0; b < SHADOW_REFINE_STEPS; ++b) {
                float mid = 0.5 * (lo + hi);
                float mb, mm;
                float dm = rmShadowField(node, ro + rd * mid, material, mb, mm);
                if (isnan(dm)) break;
                if (dm > 0.0) lo = mid; else hi = mid;
            }
            hitT = 0.5 * (lo + hi);
            return true;
        }
        if (dNext <= 0.0) {
            hitT = tNext;
            return true;
        }
        t = tNext;
    }
    return false;
}

void main() {
    vec2 extent = max(pc.viewport.xy, vec2(1.0));
    vec2 ndc = gl_FragCoord.xy / extent * 2.0 - 1.0;

    // Orthographic light ray from the cascade near plane to the far plane.
    vec4 nearH = pc.shadowInvLightSpace * vec4(ndc, 0.0, 1.0);
    vec4 farH = pc.shadowInvLightSpace * vec4(ndc, 1.0, 1.0);
    if (abs(nearH.w) < 1e-8 || abs(farH.w) < 1e-8) discard;
    vec3 ro = nearH.xyz / nearH.w;
    vec3 rd = normalize(farH.xyz / farH.w - ro);
    if (!(dot(rd, rd) > 0.5)) discard;

    // World size of one shadow texel from the inverse ortho matrix: a full
    // NDC span (2 units) maps to 2 * |invLightSpace column| world units.
    float texelWorld = max(
        max(2.0 * length(pc.shadowInvLightSpace[0].xyz), 1.0e-6) / extent.x,
        max(2.0 * length(pc.shadowInvLightSpace[1].xyz), 1.0e-6) / extent.y);

    vec3 invRd = vec3(rmInvComp(rd.x), rmInvComp(rd.y), rmInvComp(rd.z));
    float nearest = 1.0e30;
    bool found = false;
    for (uint i = 0u; i < pc.rootCount; ++i) {
        if (rmRoots[i].kind != 0u) continue; // water is not an opaque caster
        float t;
        if (rmShadowMarchRoot(rmRoots[i].node, ro, rd, invRd, nearest,
                              texelWorld, t) && t < nearest) {
            nearest = t;
            found = true;
        }
    }
    if (!found) discard;

    vec3 hitPos = ro + rd * nearest;
    vec4 lsPos = ubo.viewProjection * vec4(hitPos, 1.0);
    float depth = clamp(lsPos.z / lsPos.w, 0.0, 1.0);
    gl_FragDepth = depth;
    outEVSM = evsmMoments(depth);
}

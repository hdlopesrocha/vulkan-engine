#version 450

// Grass-only EVSM caster for the generic SDF scene.
//
// The shadow pass rasterizes one proxy cube per GRASS container (the host
// draws only the grass container range of the merged scene). Each fragment
// marches the light ray through the container and writes the EVSM moments of
// the nearest GRASS surface hit, using the exact ShadowRenderer.frag
// convention: vec2(exp(2*z), exp(4*z)) with z = clamped light-space depth of
// the hit; z = 1 (the far plane) when no grass is hit.
//
// GRASS-ONLY GUARANTEE: the candidate loop skips every definition whose prim
// is not SDF_PRIM_GRASS before evaluation, and the host only rasterizes the
// grass container range; flames, smoke, rocks and every other SDF type have
// no path to outEVSM.
//
// Wind note: the shared wind field lives at set=0 binding 27. The required
// pipeline layout carries the SDF set at index 1 + push constants only (the
// shadow pass has its own cascade set bound at set 0), so this shader marches
// the clump in its rest pose. The CPU instance AABB already covers the full
// lean budget, so the shadow volume stays conservative; only the animated
// lean is not reproduced. params.x carries the SDF clock so the phase stays
// available if the wind field is ever bound to this pass.

#include "../../types/SdfContainer.glsl"
#include "../../types/SdfDefinition.glsl"
#include "../../types/SdfGridCell.glsl"
#include "../../types/SdfInstance.glsl"
#include "../../types/SdfPrimitiveType.glsl"
#include "../../includes/Locations.glsl"

layout(location = VARY_POSWORLD) in vec3 fragWorldPos;
layout(location = VARY_BRUSHPATCH) flat in int fragContainerIndex;

layout(location = FRAG_OUT_COLOR) out vec2 outEVSM;

#include "../../includes/sdf/SdfOps.glsl"
#include "../../includes/sdf/SdfModel.glsl"
#include "../../includes/sdf/SdfGrass.glsl"

layout(std430, set = 1, binding = 0) readonly buffer SdfInstanceBuffer {
    SdfInstance sdfInstances[];
};
layout(std430, set = 1, binding = 1) readonly buffer SdfDefinitionBuffer {
    SdfDefinition sdfDefinitions[];
};
layout(std430, set = 1, binding = 3) readonly buffer SdfContainerBuffer {
    SdfContainer sdfContainers[];
};
layout(std430, set = 1, binding = 4) readonly buffer SdfGridCellBuffer {
    SdfGridCell sdfGridCells[];
};
layout(std430, set = 1, binding = 5) readonly buffer SdfGridIndexBuffer {
    uint sdfGridIndices[];
};

// Push constants (112 B; C++ twin SdfGrassShadowPC in SdfRenderer.hpp).
layout(push_constant) uniform SdfGrassShadowPC {
    mat4 lightViewProj; // cascade light view-projection (world -> light clip)
    vec4 params;        // x = time (s), y = max steps, z = epsilon, w = safety
    vec4 march;         // x = max step (m), y = min step (m), z = shadow LOD camScale
    vec4 lightDir;      // xyz = light-to-scene direction (world), w unused
} pc;

const int SDF_GRASS_SHADOW_MAX_STEPS_HARD = 256;
const uint SDF_GRASS_MAX_CANDIDATES = 8u;

// One grass clump, in the same local frame as sdfEvalInstance
// (SdfRenderer.frag): the generic SdfModel supplies the inverse TRS and the
// conservative local->world distance scale.
float sdfGrassShadowEval(vec3 wpos, SdfInstance inst, SdfDefinition def, float lodCamScale) {
    float ds;
    vec3 q = sdfModelToLocal(sdfModelFromInstance(inst), wpos, ds);
    // Shadow LOD (see the wind note above): the march evaluates the reduced
    // blade set for pc.march.z (shadow LOD camScale) instead of the full
    // clump; beyond the coarse threshold it falls back to the aggregate
    // envelope, which is the cheapest shadow representation.
    return sdGrassClump(q, def.params0, def.params1, inst.seed, lodCamScale,
                        vec2(0.0), 0.0) * ds;
}

void main() {
    if (fragContainerIndex < 0 || fragContainerIndex >= int(sdfContainers.length())) discard;
    SdfContainer cont = sdfContainers[fragContainerIndex];
    vec3 bMin = cont.boundsMin;
    vec3 bMax = cont.boundsMax;
    uvec3 dim = uvec3(max(cont.resX, 1u), max(cont.resY, 1u), max(cont.resZ, 1u));
    uint cellBase = cont.cellStart;

    vec3 rd = pc.lightDir.xyz;
    float rd2 = dot(rd, rd);
    if (rd2 < 1e-8) discard;
    rd *= inversesqrt(rd2);

    // Ray interval inside the container. The proxy front face is the entry
    // point; the AABB slab maximum is the exit. Back-face fragments (the
    // proxy is drawn with cull NONE) start outside the slab and exit
    // immediately (tExit <= 0): they fall through to the far-depth write.
    vec3 invDir = vec3(
        (abs(rd.x) > 1e-12) ? (1.0 / rd.x) : 1e12,
        (abs(rd.y) > 1e-12) ? (1.0 / rd.y) : 1e12,
        (abs(rd.z) > 1e-12) ? (1.0 / rd.z) : 1e12);
    vec3 t0 = (bMin - fragWorldPos) * invDir;
    vec3 t1 = (bMax - fragWorldPos) * invDir;
    float tExit = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));

    int maxSteps = int(pc.params.y + 0.5);
    if (maxSteps <= 0) maxSteps = 64;
    maxSteps = min(maxSteps, SDF_GRASS_SHADOW_MAX_STEPS_HARD);
    float eps = (pc.params.z > 0.0) ? pc.params.z : 0.01;
    float safety = clamp((pc.params.w > 0.0) ? pc.params.w : 0.7, 0.1, 1.0);
    float minStep = max(pc.march.y, 1e-4);
    float maxStep = max((pc.march.x > 0.0) ? pc.march.x : 1.0, minStep);

    vec3 cellSize = max((bMax - bMin) / vec3(dim), vec3(1e-6));
    int nInstances = int(sdfInstances.length());
    int nDefs = int(sdfDefinitions.length());
    int nCells = int(sdfGridCells.length());
    int nIndices = int(sdfGridIndices.length());

    float t = 0.0;
    bool hit = false;
    float hitT = 0.0;

    for (int i = 0; i < SDF_GRASS_SHADOW_MAX_STEPS_HARD; ++i) {
        if (i >= maxSteps || t >= tExit) break;
        vec3 p = fragWorldPos + rd * t;

        // Uniform-grid DDA (same traversal as SdfRenderer.frag): the cell
        // lists the clump candidates; empty cells jump to the cell exit.
        vec3 fpos = clamp((p - bMin) / cellSize, vec3(0.0), vec3(dim) - vec3(1e-4));
        ivec3 cell = clamp(ivec3(floor(fpos)), ivec3(0), ivec3(dim) - ivec3(1));
        vec3 cellMin = bMin + vec3(cell) * cellSize;
        vec3 cellMax = cellMin + cellSize;
        vec3 tMaxV = vec3(
            (abs(rd.x) > 1e-12) ? (((rd.x > 0.0) ? cellMax.x : cellMin.x) - p.x) / rd.x : 1e5,
            (abs(rd.y) > 1e-12) ? (((rd.y > 0.0) ? cellMax.y : cellMin.y) - p.y) / rd.y : 1e5,
            (abs(rd.z) > 1e-12) ? (((rd.z > 0.0) ? cellMax.z : cellMin.z) - p.z) / rd.z : 1e5);
        float dtCell = max(min(min(tMaxV.x, tMaxV.y), tMaxV.z), 0.0);

        uint lin = uint((cell.z * int(dim.y) + cell.y) * int(dim.x) + cell.x);
        uint cellAddr = cellBase + lin;
        uint coff = 0u;
        uint ccnt = 0u;
        if (cellAddr < uint(nCells)) {
            coff = sdfGridCells[cellAddr].offset;
            ccnt = sdfGridCells[cellAddr].count;
        }
        if (ccnt == 0u) {
            t += max(dtCell, minStep) + 1e-4;
            continue;
        }

        uint n = min(ccnt, SDF_GRASS_MAX_CANDIDATES);
        float dBest = 1e5;
        float nearestBox = 1e5;
        bool haveBest = false;
        for (uint k = 0u; k < SDF_GRASS_MAX_CANDIDATES; ++k) {
            if (k >= n) break;
            uint gAddr = coff + k;
            if (gAddr >= uint(nIndices)) break;
            uint ii = sdfGridIndices[gAddr];
            if (ii >= uint(nInstances)) continue;
            SdfInstance inst = sdfInstances[ii];
            uint di = inst.defIdx;
            if (di >= uint(nDefs)) continue;
            SdfDefinition dd = sdfDefinitions[di];
            // Grass-only guarantee: every non-grass primitive is skipped
            // before evaluation and can never contribute a distance, a hit
            // or a march step.
            if (dd.prim != SDF_PRIM_GRASS) continue;
            vec3 oq = max(max(inst.boundsMin - p, p - inst.boundsMax), vec3(0.0));
            float ob = length(oq);
            float skipR = maxStep + 0.1;
            if (ob > skipR) {
                nearestBox = min(nearestBox, ob);
                continue;
            }
            float d = sdfGrassShadowEval(p, inst, dd, pc.march.z);
            // Same combination rule as sdfEvalInstance/SdfRenderer.frag, so
            // the marched field matches the rendered clump union exactly
            // (smooth-union creases included; never larger than the union).
            float kk = clamp(dd.smoothK, 0.0, 2.0);
            if (!haveBest) { dBest = d; haveBest = true; }
            else { dBest = sdfCombine(dBest, d, dd.op, kk); }
        }
        if (!haveBest) {
            // Conservative advance to the nearest rejected clump AABB, capped
            // by the cell exit (same rule as SdfRenderer.frag).
            float jump = min(max(dtCell, 0.0), max(nearestBox, minStep));
            t += max(jump, minStep) + 1e-4;
            continue;
        }
        if (dBest < eps) {
            hit = true;
            hitT = t;
            break;
        }
        // Grass step law (mirrors SdfRenderer.frag): pure sphere tracing with
        // no scale floor, so the march can never tunnel a thin blade.
        float dt = min(dBest * safety, maxStep);
        t += max(dt, 1e-5);
    }

    // Nearest hit -> light-space depth (ShadowRenderer.frag convention);
    // no hit -> the far plane so empty texels keep the cleared far moments.
    float z = 1.0;
    if (hit) {
        vec4 lsPos = pc.lightViewProj * vec4(fragWorldPos + rd * hitT, 1.0);
        if (lsPos.w > 1e-6) z = clamp(lsPos.z / lsPos.w, 0.0, 1.0);
    }
    outEVSM = vec2(exp(2.0 * z), exp(4.0 * z));
    // Depth test against the rasterized casters already in the depth buffer:
    // grass behind a solid/vegetation occluder must not overwrite its nearer
    // moments, and grass in front must overwrite the farther ones.
    gl_FragDepth = z;
}

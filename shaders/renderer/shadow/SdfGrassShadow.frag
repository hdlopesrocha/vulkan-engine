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
// M10 (report 27): march scalars derived once per vertex into flat varyings
// (identical ops to the old per-fragment derivation, which is deleted
// below); flat interpolation carries them unchanged to every fragment.
layout(location = 16) flat in vec4 vShadowMarch; // x=eps, y=safety, z=minStep, w=maxStep
layout(location = 17) flat in int vShadowMaxSteps;

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

// Push constants (128 B; C++ twin SdfGrassShadowPC in SdfRenderer.hpp,
// shared by the full and the SHADOW_IMPOSTOR_ONLY variant pipelines).
layout(push_constant) uniform SdfGrassShadowPC {
    mat4 lightViewProj; // cascade light view-projection (world -> light clip)
    vec4 params;        // x = time (s), y = max steps, z = epsilon, w = safety
    vec4 march;         // x = max step (m), y = min step (m), z = shadow LOD camScale
    vec4 impostor;      // x = impostor fade start, y = fade full (clump scales)
    vec4 lightDir;      // xyz = light-to-scene direction (world), w unused
} pc;

// ── Step-budget proof (report 27 L15; main-march twins at SdfRenderer.frag:138-139) ──
// SDF_GRASS_SHADOW_MAX_STEPS_HARD = 256 is the compiler-visible loop bound
// only; the EFFECTIVE budget is pc.params.y (from SdfParamsUBO::maxSteps,
// default 64), clamped in the vertex stage. SDF_GRASS_MAX_CANDIDATES = 8
// matches the grid cell storage cap, mirroring the main march's first-8 rule.
// The grass step law below (:244-247) is pure sphere tracing on EXACT
// distances: the clump lean is rigid (rotations preserve distances; this
// pass marches the rest pose, see the wind note above) and grass carries no
// noise/spike deformation, so no Lipschitz halving applies and a step of
// dBest * safety (safety < 1) can never overstep the field. The maxStep
// clamp only shortens longer jumps; the min advance (1e-5) plus the
// rejected-AABB advance (nearestBox, same rule as SdfRenderer.frag) guarantee
// progress every iteration. Budget rule: max overstep per step =
// maxStep * safety must stay below the min blade width at the shadow LOD
// (pc.march.z; the SHADOW_IMPOSTOR_ONLY variant marches zero blades, so its
// bound is the impostor cone, looser still). Keep this proof when retuning:
// raising maxStep past the blade width breaks sphere-tracing conservatism and
// tunnels thin blades. Comment only; zero code change.
const int SDF_GRASS_SHADOW_MAX_STEPS_HARD = 256;
const uint SDF_GRASS_MAX_CANDIDATES = 8u;

// H8 impostor-only shadow representation (SHADOW_IMPOSTOR_ONLY=1 pipeline
// variant): the hard far-LOD body of sdGrassClump (base-bulged round cone +
// ripple, zero blades) in the shadow rest pose (up = +Y, no lean — the shadow
// pass never binds the wind field, see the wind note above). Constants mirror
// sdGrassClump (bend ceiling 1.4, base bulge 0.5, ripple 0.30, freq 2.5,
// detail 0.35 x height / 6.0 x width); keep in sync with SdfGrass.glsl
// (TODO: factor one shared helper). Matches the full evaluator at
// camScale >= impostorFull in the rest pose, so the EVSM moment contract
// (vec2(exp(2z), exp(4z)) + gl_FragDepth) is unchanged.
#ifdef SHADOW_IMPOSTOR_ONLY
float sdGrassShadowImpostorOnly(vec3 p, vec4 p0, vec4 p1, float seed) {
    float radius = max(p0.x, 1e-3);
    float height = max(p0.y, 1e-3);
    float curve = clamp(p1.x, 0.0, 1.5);
    float width = clamp(p0.z, 1e-5,
                        min(max(radius * 0.5, 1e-4), max(height * 0.25, 1e-4)));
    vec3 up = vec3(0.0, 1.0, 0.0);
    float envR0 = radius + 2.0 * width;
    float envR1 = radius + 2.0 * width + height * curve * 1.4;
    float impScale = max(radius, height * 0.5);
    float impAmp = 0.30 * impScale;
    float impFreq = 2.5 / max(impScale, 1e-3);
    float impR0 = envR0 + 0.5 * (envR1 - envR0);
    float impR1 = envR1;
    float impLip = sdGrassImpostorLip(height, impAmp, impFreq);
    float icd = sdGrassRoundCone(p, vec3(0.0), up * height, impR0, impR1);
    float detail = max(height * 0.35, width * 6.0);
    if (icd > max(detail, impAmp * 1.5)) return (icd - impAmp) / impLip;
    return sdGrassImpostor(p, up, height, icd, seed, impAmp, impFreq);
}
#endif

// One grass clump, in the same local frame as sdfEvalInstance
// (SdfRenderer.frag): the generic SdfModel supplies the inverse TRS and the
// conservative local->world distance scale.
float sdfGrassShadowEval(vec3 wpos, SdfInstance inst, SdfDefinition def, float lodCamScale) {
    float ds;
    vec3 q = sdfModelToLocal(sdfModelFromInstance(inst), wpos, ds);
#ifdef SHADOW_IMPOSTOR_ONLY
    // H8: outer cascades never resolve blades — the blade loop in sdGrassClump
    // is compiled out instead of sitting behind the pc.march.z runtime branch.
    // (lodCamScale intentionally unused: the impostor body has no blade LOD.)
    return sdGrassShadowImpostorOnly(q, def.params0, def.params1, inst.seed) * ds;
#else
    // Shadow LOD (see the wind note above): the march evaluates the reduced
    // blade set for pc.march.z (shadow LOD camScale) instead of the full
    // clump. At pc.march.z >= GRASS_IMPOSTOR_FULL (80, SdfGrass.glsl) the
    // impostor is the whole field: zero blades are evaluated and the caster
    // marches one base-bulged round cone + ripple, the cheapest shadow
    // representation. Between GRASS_IMPOSTOR_START (48) and FULL the reduced
    // blades and the fading impostor are unioned; below START it is the
    // blade LOD only.
    return sdGrassClump(q, def.params0, def.params1, inst.seed, lodCamScale,
                        vec2(0.0), 0.0, pc.impostor.x, pc.impostor.y) * ds;
#endif
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

    // M10 (report 27): march scalars arrive from the vertex stage (flat, so
    // every fragment sees the identical value the old per-fragment
    // derivation produced there).
    int maxSteps = vShadowMaxSteps;
    float eps = vShadowMarch.x;
    float safety = vShadowMarch.y;
    float minStep = vShadowMarch.z;
    float maxStep = vShadowMarch.w;

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
            // C2 (report 27): AABB test ABOVE the definition fetch — it needs
            // only the instance bounds, so AABB-rejected candidates (the
            // common case, the reason skipR exists) skip the SdfDefinition
            // SSBO load entirely. The prim test stays after the fetch (prim
            // lives in the definition).
            vec3 oq = max(max(inst.boundsMin - p, p - inst.boundsMax), vec3(0.0));
            float ob = length(oq);
            float skipR = maxStep + 0.1;
            if (ob > skipR) {
                nearestBox = min(nearestBox, ob);
                continue;
            }
            uint di = inst.defIdx;
            if (di >= uint(nDefs)) continue;
            SdfDefinition dd = sdfDefinitions[di];
            // Grass-only guarantee: every non-grass primitive is skipped
            // before evaluation and can never contribute a distance, a hit
            // or a march step.
            if (dd.prim != SDF_PRIM_GRASS) continue;
            float d = sdfGrassShadowEval(p, inst, dd, pc.march.z);
            // C2 (report 27): plain-union fast path — every grass definition
            // uses op=Union (SdfScene::createGrassFromAnchors) and plain
            // union provably reduces to min() (sdfCombine falls through to
            // opUnion; smoothK is only read by the SMOOTH_* arms), so union
            // combines cost one min. Other ops keep the sdfCombine dispatch.
            // skipR/nearestBox/haveBest ordering untouched.
            float kk = clamp(dd.smoothK, 0.0, 2.0);
            if (!haveBest) { dBest = d; haveBest = true; }
            else if (dd.op == SDF_OP_UNION) { dBest = min(dBest, d); }
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

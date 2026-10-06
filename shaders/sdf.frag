#version 450

#include "ubo/BulletGPU.glsl"
#include "ubo/SdfContainerGPU.glsl"
#include "ubo/SdfDefinitionGPU.glsl"
#include "ubo/SdfGridCellGPU.glsl"
#include "ubo/SdfInstanceGPU.glsl"
#include "ubo/SdfMaterialGPU.glsl"
#include "ubo/SdfParamsUBO.glsl"
#include "types/SmokeBulletState.glsl"
#include "types/SmokeSample.glsl"

// Generic SDF raymarcher: surface + volume + emissive + transparent modes
// sharing one traversal. Container proxy -> ray/AABB -> scene-depth clamp ->
// uniform-grid DDA (empty-space skipping) -> candidate AABB test ->
// combined SDF (adaptive stepping) -> density/temperature/emission
// front-to-back accumulation with early termination.

#include "includes/locations.glsl"

layout(location = VARY_POSWORLD) in vec3 fragWorldPos;
layout(location = VARY_BRUSHPATCH) flat in int fragContainerIndex;

#include "includes/ubo.glsl"
#include "includes/sdf_material.glsl"
#include "includes/sdf_primitives.glsl"
#include "includes/sdf_ops.glsl"
#include "includes/sdf_noise.glsl"
#include "includes/sdf_smoke.glsl"
// Shared wind field (set 0, binding 27) for the flame lean below. Requires
// perlin.glsl (perlinNoise3D) before wind_field.glsl. Set 0 is the global
// scene set, already first in this pipeline's layout (SdfRenderer), whose
// binding 27 carries VERTEX|FRAGMENT|COMPUTE stage flags — no C++ change.
#include "includes/perlin.glsl"
#include "includes/wind_field.glsl"

layout(std430, set = 1, binding = 0) readonly buffer SdfInstanceBuffer {
    SdfInstanceGPU sdfInstances[];
};
layout(std430, set = 1, binding = 1) readonly buffer SdfDefinitionBuffer {
    SdfDefinitionGPU sdfDefinitions[];
};
layout(std430, set = 1, binding = 2) readonly buffer SdfMaterialBuffer {
    SdfMaterialGPU sdfMaterials[];
};
layout(std430, set = 1, binding = 3) readonly buffer SdfContainerBuffer {
    SdfContainerGPU sdfContainers[];
};
layout(std430, set = 1, binding = 4) readonly buffer SdfGridCellBuffer {
    SdfGridCellGPU sdfGridCells[];
};
layout(std430, set = 1, binding = 5) readonly buffer SdfGridIndexBuffer {
    uint sdfGridIndices[];
};
layout(set = 1, binding = 6) uniform SdfParamsBlock {
    SdfParamsUBO sdfParams;
};
layout(set = 1, binding = 7) uniform sampler2D sdfSceneDepth;

layout(location = FRAG_OUT_COLOR) out vec4 outColor;

const int SDF_MAX_STEPS_HARD = 256;
const uint SDF_MAX_CANDIDATES = 8u;

bool sdfRayAabb(vec3 ro, vec3 rd, vec3 bMin, vec3 bMax,
                out float tEnter, out float tExit) {
    vec3 invDir = vec3(
        (abs(rd.x) > 1e-12) ? (1.0 / rd.x) : 1e12,
        (abs(rd.y) > 1e-12) ? (1.0 / rd.y) : 1e12,
        (abs(rd.z) > 1e-12) ? (1.0 / rd.z) : 1e12);
    vec3 t0 = (bMin - ro) * invDir;
    vec3 t1 = (bMax - ro) * invDir;
    vec3 tSmaller = min(t0, t1);
    vec3 tBigger = max(t0, t1);
    tEnter = max(max(tSmaller.x, tSmaller.y), tSmaller.z);
    tExit = min(min(tBigger.x, tBigger.y), tBigger.z);
    return tExit > max(tEnter, 0.0);
}

float sdfSceneDistance(vec3 ro, vec2 uv) {
    vec2 cuv = clamp(uv, vec2(0.0), vec2(1.0));
    float raw = textureLod(sdfSceneDepth, cuv, 0.0).r;
    if (raw >= 1.0) return 1e5;
    vec4 w = ubo.invViewProjection * vec4(cuv * 2.0 - 1.0, raw, 1.0);
    if (abs(w.w) < 1e-8) return 1e5;
    return distance(ro, w.xyz / w.w);
}

// World -> local (inverse rigid + anisotropic radius/height scales).
// Returns local point; outScale carries the conservative distance scale
// (uniform * min(radius, height)) so sphere tracing never oversteps.
vec3 sdfWorldToLocal(vec3 wpos, SdfInstanceGPU inst, out float outScale) {
    float uni = max(inst.posScale.w, 1e-4);
    float rSc = max(inst.sizeParams.y, 1e-4);
    float hSc = max(inst.sizeParams.x, 1e-4);
    vec3 q = sdfTransformInverse(wpos, inst.posScale.xyz, inst.rotSeed.xyz, uni);
    q.x /= rSc;
    q.z /= rSc;
    q.y /= hSc;
    outScale = uni * min(rSc, hSc);
    return q;
}

float sdfEvalInstance(vec3 wpos, SdfInstanceGPU inst, SdfDefinitionGPU def,
                      SdfMaterialGPU mat, float time) {
    // Smoke volumes march their grown radius + bullet-carved field
    // (sdf_smoke.glsl); all other primitives share the path below.
    if (def.meta.x == SDF_PRIM_SMOKE) {
        return smokeMarchSDF(wpos, inst, def, time);
    }
    float ds;
    vec3 q = sdfWorldToLocal(wpos, inst, ds);
    // Optional repeat before primitive eval (deform bit4): period from params1.xyz.
    uint deform = def.meta.z;
    if ((deform & 16u) != 0u) {
        q = opRepeat(q, abs(def.params1.xyz));
    }
    float d = sdfPrimitive(q, def.meta.x, def.params0, def.params1);
    // Deformation runs in canonical flame space (flame ~3.2 units tall) so
    // waviness/spike SIZE stays constant under instance scaling; in raw
    // local units an x32 flame would get x32-stretched blobby features.
    // (Offsets stay in local units, consistent with d.)
    vec3 qn = q;
    float hhn = 3.2;
    if (def.meta.x == SDF_PRIM_FLAME) {
        float hh = max(def.params0.y, 1e-3);
        qn = q * (3.2 / hh);
        // Shared-field flame lean: displace the canonical flame domain by
        // the world-space field (XZ) at the instance position on the SDF
        // clock, so both sdfFlameDeform and sdfFlameSpikes below sample
        // wind-bent input and flames lean downwind. Gain 0.05 canonical
        // units per (m/s), scaled by (0.25 + turbulence) so the response
        // tracks the existing turbulence setting (rise is a scroll rate, not
        // a displacement gain, and is left untouched); clamped to +/-0.5 so
        // tornado cores cannot throw samples out of the grid. Evaluated only
        // when a consumer below is active (deform bit or spikes), mirroring
        // their exact conditions, so undeformed flames pay zero extra ALU.
        // NaN-safe: windSVF is NaN-safe and the clamp bounds the shift.
        float spkAmp = max(def.params1.y, 0.0);
        if (((deform & 1u) != 0u) || (spkAmp > 0.001)) {
            float turbResp = 0.25 + clamp(mat.extra.z, 0.0, 2.0);
            vec3 wfFire = windSVF(wpos, time);
            vec2 fireLean = clamp(wfFire.xz * (0.05 * turbResp), vec2(-0.5), vec2(0.5));
            qn.x += fireLean.x;
            qn.z += fireLean.y;
        }
    }
    // Procedural deformation offsets (noise + spikes) are not metric: adding
    // them can push |grad d| above 1 by roughly (deform amplitude / feature
    // size), so the returned distance may overestimate the true distance and
    // sphere tracing would skip through the field. Flag every deformed
    // definition and compensate below; the undeformed path is unchanged.
    bool deformed = false;
    if ((deform & 1u) != 0u) {
        float turb = mat.extra.z;
        float rise = mat.extra.w;
        d += sdfFlameDeform(qn, time, inst.rotSeed.w, turb, rise);
        deformed = true;
    }
    if (def.meta.x == SDF_PRIM_FLAME) {
        // Tapered-flame spikes (params1.y = amplitude, .z = frequency):
        // ridged tongues over the smooth capsule; 0 = rounded capsule.
        float spk = max(def.params1.y, 0.0);
        if (spk > 0.001) {
            d += sdfFlameSpikes(qn, hhn, inst.rotSeed.w, def.params1.z, spk);
            deformed = true;
        }
    }
    // Conservative Lipschitz compensation for noise/spike-deformed fields:
    // halve the canonical distance so the world distance (d * ds) can no
    // longer overestimate the true distance by more than the deform gain.
    // A uniform factor cancels in sdfSurfaceNormal (normalize), so shading
    // is unaffected; only the march step bound becomes conservative.
    if (deformed) {
        d *= 0.5;
    }
    return d * ds;
}

vec3 sdfSurfaceNormal(vec3 p, SdfInstanceGPU inst, SdfDefinitionGPU def,
                      SdfMaterialGPU mat, float time, float e) {
    vec3 k0 = vec3(1.0, -1.0, -1.0);
    vec3 k1 = vec3(-1.0, -1.0, 1.0);
    vec3 k2 = vec3(-1.0, 1.0, -1.0);
    vec3 k3 = vec3(1.0, 1.0, 1.0);
    float f0 = sdfEvalInstance(p + k0 * e, inst, def, mat, time);
    float f1 = sdfEvalInstance(p + k1 * e, inst, def, mat, time);
    float f2 = sdfEvalInstance(p + k2 * e, inst, def, mat, time);
    float f3 = sdfEvalInstance(p + k3 * e, inst, def, mat, time);
    vec3 n = k0 * f0 + k1 * f1 + k2 * f2 + k3 * f3;
    float l = length(n);
    return (l > 1e-9) ? (n / l) : vec3(0.0, 1.0, 0.0);
}

float sdfProjDepth(vec3 ro, vec3 rd, float t) {
    vec4 c = ubo.viewProjection * vec4(ro + rd * t, 1.0);
    if (c.w <= 1e-6) return 0.0;
    return clamp(c.z / c.w, 0.0, 1.0);
}

void main() {
    if (fragContainerIndex < 0 || fragContainerIndex >= sdfContainers.length()) discard;
    SdfContainerGPU cont = sdfContainers[fragContainerIndex];
    vec3 bMin = cont.boundsMin.xyz;
    vec3 bMax = cont.boundsMax.xyz;
    uvec3 dim = uvec3(max(cont.gridInfo.x, 1u), max(cont.gridInfo.y, 1u), max(cont.gridInfo.z, 1u));
    uint cellBase = cont.gridOffset.x;
    // NOTE: gridInfo.w (global index start) is already baked into each
    // cell's offset by flatten(); it must NOT be added again here.

    vec3 ro = ubo.viewPosition;
    vec3 toFrag = fragWorldPos - ro;
    float toFragLen = length(toFrag);
    if (toFragLen < 1e-8) discard;
    vec3 rd = toFrag / toFragLen;

    float tEnter, tExit;
    if (!sdfRayAabb(ro, rd, bMin, bMax, tEnter, tExit)) discard;
    tEnter = max(tEnter, 0.0);

    // Depth-clamp against the opaque scene (binding 7): the SDF task waits
    // on tlSolid, after which the solid pass has transitioned its depth to
    // SHADER_READ_ONLY_OPTIMAL, so sampling here is race-free. Fully
    // occluded rays discard; partially occluded rays march only to the
    // occluder. (Explicit LOD: divergent flow, derivatives undefined here.)
    vec4 pclip = ubo.viewProjection * vec4(fragWorldPos, 1.0);
    if (pclip.w > 1e-6) {
        vec2 suv = pclip.xy / pclip.w * 0.5 + 0.5;
        tExit = min(tExit, sdfSceneDistance(ro, suv));
    }
    if (tEnter >= tExit) discard;

    // March params from SdfParamsUBO (CPU layout): timeDebug=(t,packed,maxSteps,safety),
    // marchParams=(minStep,maxStep,epsilon,earlyTerm). Zero-filled UBO -> safe defaults.
    int maxSteps = int(sdfParams.timeDebug.z + 0.5);
    if (maxSteps <= 0) maxSteps = 64;
    maxSteps = min(maxSteps, SDF_MAX_STEPS_HARD);
    float safety = sdfParams.timeDebug.w;
    if (safety <= 0.0) safety = 0.7;
    safety = clamp(safety, 0.1, 1.0);
    float eps = sdfParams.marchParams.z;
    if (eps <= 0.0) eps = 0.01;
    float opacityThresh = sdfParams.marchParams.w;
    if (opacityThresh <= 0.0) opacityThresh = 0.99;
    opacityThresh = clamp(opacityThresh, 0.01, 1.0);
    float minStep = sdfParams.marchParams.x;
    if (minStep <= 0.0) minStep = 0.05;
    float maxStep = sdfParams.marchParams.y;
    if (maxStep <= 0.0) maxStep = 1.0;
    maxStep = max(maxStep, minStep);
    float time = sdfParams.timeDebug.x;
    uint packedDbg = uint(sdfParams.timeDebug.y + 0.5);
    uint renderMode = (packedDbg >> 16u) & 0xFFFFu; // 0 surface,1 volume,2 emissive,3 transparent
    uint debugFlags = packedDbg & 0xFFFFu;
    bool debugView = (debugFlags & 1u) != 0u;

    vec3 cellSize = max((bMax - bMin) / vec3(dim), vec3(1e-6));
    float containerH = max(bMax.y - bMin.y, 1e-6);
    int nInstances = sdfInstances.length();
    int nDefs = sdfDefinitions.length();
    int nMats = sdfMaterials.length();
    int nCells = sdfGridCells.length();
    int nIndices = sdfGridIndices.length();

    float t = tEnter;
    float trans = 1.0;
    vec3 accum = vec3(0.0);
    float tFirst = -1.0;
    float tDepthAccum = 0.0; // transmittance-weighted ray distance (flame centroid)
    int steps = 0;
    int hits = 0;          // steps where >=1 candidate SDF was evaluated
    float minAbsD = 1e5;   // closest approach to any flame surface this ray
    // Last-sample smoke metrics (§24 debug views 1-9).
    float dbgSmokeSDF = 1e5;
    float dbgSmokeDens = 0.0;
    float dbgSmokeBullet = 1e5;
    float dbgSmokeTunnel = 0.0;
    float dbgSmokePress = 0.0;
    float dbgSmokeWave = 0.0;
    float dbgSmokeTurb = 0.0;
    float dbgSmokeWake = 0.0;
    float dbgSmokeFinal = 0.0;
    float dbgSmokeLight = 0.0;
    bool hit = false;
    float hitT = tEnter;
    vec3 hitColor = vec3(0.0);

    // Visible tracer round: analytic gold capsule proxy, evaluated ONCE
    // before the march (the old per-sample sphere test aliased into stacked
    // horizontal discs). Each live in-flight bullet is approximated by K=8
    // overlapping spheres tapering from tail to nose; exact ray/sphere
    // roots give a banding-free hit distance. The tail is floored at 40% of
    // the nose so a widening cone still reads as ONE continuous slug
    // instead of two detached balls. The smoke container expands over the
    // flight path, so the proxy is covered even outside the smoke ball.
    float bestT = 1e5;
    vec3 bestC = vec3(0.0);
    float bestR = 0.0;
    for (int bi = 0; bi < 8; ++bi) {
        BulletGPU bbl = smokeBullets[bi];
        if (bbl.c.z <= 0.0) continue;
        SmokeBulletState bst = smokeBulletState(bbl, time);
        if (!bst.live || !bst.headOnPath) continue;
        vec3 bD = bbl.b.xyz / max(length(bbl.b.xyz), 1e-6);
        vec3 head = bbl.a.xyz + bD * bst.traveled;
        float noseR = max(bbl.c.x * 0.30, 0.3);
        float tailR = max(max(bbl.a.w * 0.30, noseR * 0.4), 0.5);
        // Round length follows the bigger end so a widening cone (small
        // launch bore, huge head bore) renders as one growing round.
        float L = max(max(bbl.a.w, bbl.c.x), 1.0);
        vec3 tail = head - bD * (L * 0.7);
        vec3 nose = head + bD * (L * 0.3);
        for (int k = 0; k < 8; ++k) {
            float fk = float(k) / 7.0;
            vec3 center = mix(tail, nose, fk);
            float radius = mix(tailR, noseR, fk);
            vec3 oc = ro - center;
            float bq = dot(oc, rd);
            float cq = dot(oc, oc) - radius * radius;
            float disc = bq * bq - cq;
            if (disc > 0.0) {
                float tHit = -bq - sqrt(disc);
                if (tHit > 0.0 && tHit < bestT) {
                    bestT = tHit;
                    bestC = center;
                    bestR = radius;
                }
            }
        }
    }

    for (int i = 0; i < SDF_MAX_STEPS_HARD; i++) {
        if (i >= maxSteps || t > tExit) break;
        steps = i + 1;
        vec3 p = ro + rd * t;

        // Analytic tracer hit (proxy computed before the loop): stop at the
        // exact ray/sphere entry and shade the gold round attenuated by the
        // smoke accumulated so far (front-to-back correct via trans).
        if (bestT < 1e4 && t >= bestT) {
            vec3 hp = ro + rd * bestT;
            vec3 bN = (hp - bestC) / max(bestR, 1e-4);
            vec3 tang = normalize(abs(bN.y) < 0.99 ? cross(bN, vec3(0.0, 1.0, 0.0)) : cross(bN, vec3(1.0, 0.0, 0.0)));
            float e0 = sdfNoise(hp * 2.0 + bestC);
            float e1 = sdfNoise(hp * 2.0 + bestC + vec3(4.7));
            // Normal distortion gain is widget-controlled: guard the normalize
            // so a cancelling perturbation can never divide by zero.
            vec3 bNp = bN + (tang * (e0 - 0.5) + cross(bN, tang) * (e1 - 0.5))
                            * max(smokeTuning.gold2.w, 0.0);
            vec3 bNt = (dot(bNp, bNp) > 1e-12) ? normalize(bNp) : bN;
            float pat = sdfNoise(hp * max(smokeTuning.gold1.w, 0.0) + bestC);
            vec3 V = -rd;
            vec3 L = -normalize(ubo.lightDirection);
            float dif = max(dot(bNt, L), 0.0);
            // Guarded half-vector: L + V degenerates when the view direction
            // is (anti)parallel to the light. The base is clamped to [0,1]
            // so the power argument can never go negative; a non-positive
            // specular power disables the lobe instead of hitting pow(0, 0).
            vec3 H = L + V;
            float hl = length(H);
            float specDot = (hl > 1e-4) ? clamp(dot(bNt, H / hl), 0.0, 1.0) : 0.0;
            float specPower = smokeTuning.gold0.w;
            float spec = (specPower > 1e-3) ? pow(specDot, specPower) : 0.0;
            float fres = pow(clamp(1.0 - max(dot(bNt, V), 0.0), 0.0, 1.0), 3.0);
            vec3 gold = mix(smokeTuning.gold0.rgb, smokeTuning.gold1.rgb,
                            clamp(pat * 0.65 + fres * smokeTuning.gold2.y, 0.0, 1.0));
            vec3 goldCol = gold * (ubo.lightColor * (0.25 + 0.9 * dif)
                                   + vec3(1.0, 0.72, 0.25) * smokeTuning.gold2.z)
                         + ubo.lightColor * spec * smokeTuning.gold2.x;
            outColor = vec4(accum + trans * goldCol, 1.0);
            gl_FragDepth = sdfProjDepth(ro, rd, bestT);
            return;
        }

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
        uint coff = 0u, ccnt = 0u;
        if (cellAddr < uint(nCells)) {
            // Cell offsets are already global (flatten() rebases each
            // container's cells into the shared index buffer); adding
            // indexBase again would double-count and read out of range,
            // silently emptying every non-first container.
            coff = sdfGridCells[cellAddr].offset;
            ccnt = sdfGridCells[cellAddr].count;
        }
        if (ccnt == 0u) {
            // Empty-space skipping: one DDA jump to the cell exit, with a
            // guaranteed minimum progress. A ray sitting exactly on a cell
            // boundary face (dtCell == 0 there) would otherwise advance only
            // the 1e-4 epsilon per iteration and never leave the boundary
            // cell inside the step budget — that froze every ray leaving the
            // camera's cell through the boundary planes (the half-screen
            // "no smoke" cross artifact when the camera sits on a grid wall).
            t = t + max(dtCell, minStep) + 1e-4;
            continue;
        }

        uint n = min(ccnt, SDF_MAX_CANDIDATES);
        float dBest = 1e5;
        float dMin = 1e5;
        float nearestBox = 1e5; // distance to the closest rejected AABB
        SdfInstanceGPU bestInst;
        SdfDefinitionGPU bestDef;
        SdfMaterialGPU bestMat;
        bool haveBest = false;
        for (uint k = 0u; k < SDF_MAX_CANDIDATES; k++) {
            if (k >= n) break;
            uint gAddr = coff + k;
            if (gAddr >= uint(nIndices)) break;
            uint ii = sdfGridIndices[gAddr];
            if (ii >= uint(nInstances)) continue;
            SdfInstanceGPU inst = sdfInstances[ii];
            uint di = inst.indices.x;
            uint mi = inst.indices.y;
            if (di >= uint(nDefs) || mi >= uint(nMats)) continue;
            SdfMaterialGPU m0 = sdfMaterials[mi];
            vec3 oq = max(max(inst.boundsMin.xyz - p, p - inst.boundsMax.xyz), vec3(0.0));
            float ob = length(oq); // exact distance to the instance AABB
            float skipR = maxStep + 0.1 + abs(m0.extra.z);
            if (ob > skipR) {
                nearestBox = min(nearestBox, ob); // conservative advance source
                continue;
            }
            SdfDefinitionGPU dd = sdfDefinitions[di];
            float d = sdfEvalInstance(p, inst, dd, m0, time);
            float kk = clamp(sdfUnpackSmoothK(dd), 0.0, 2.0);
            if (!haveBest) { dBest = d; haveBest = true; }
            else { dBest = sdfCombine(dBest, d, dd.meta.y, kk); }
            if (d < dMin) { dMin = d; bestInst = inst; bestDef = dd; bestMat = m0; }
        }
        if (!haveBest) {
            // Every candidate was out of bounds: advance by the conservative
            // distance to the nearest instance AABB (bounded by the DDA cell
            // exit) instead of by the cell exit alone. The old full-cell jump
            // could overshoot a smoke volume whose bounds are inside a huge
            // cell, and on an exact cell-boundary start it froze the march
            // (dtCell == 0 -> no progress within the step budget), producing
            // the crossed half-plane holes. The max(..., minStep) keeps a
            // guaranteed minimum advance in every case.
            float jump = min(max(dtCell, 0.0), max(nearestBox, minStep));
            t = t + max(jump, minStep) + 1e-4;
            continue;
        }
        hits++;
        minAbsD = min(minAbsD, abs(dBest));

        // Surface mode (or any mode hitting the zero crossing): shade opaque hit.
        if (dBest < eps && (renderMode == 0u || renderMode == 3u)) {
            float e = max(eps * 2.0, 0.004);
            vec3 nn = sdfSurfaceNormal(p, bestInst, bestDef, bestMat, time, e);
            vec3 L = -normalize(ubo.lightDirection);
            float ndl = max(dot(nn, L), 0.0);
            float alpha = clamp(bestMat.surfaceParams.z, 0.0, 1.0);
            hitColor = bestMat.baseColor.rgb * (0.2 + ndl) * ubo.lightColor + bestMat.emission.rgb * 0.2;
            if (renderMode == 3u) {
                outColor = vec4(hitColor, alpha);
            } else {
                outColor = vec4(hitColor, 1.0);
            }
            gl_FragDepth = sdfProjDepth(ro, rd, t);
            return;
        }
        // Raw field sign before the volume clamp: negative means the sample
        // is inside a (possibly deformed) primitive. The fire branch uses it
        // for the interior step bound; smoke keeps its own step law.
        bool insideField = (dBest < 0.0);
        if (dBest < eps) {
            // Volume/emissive modes treat the interior as dense (no hard surface).
            dBest = eps;
        }

        // Scale-aware volumetric stepping (fire path): the SDF distance
        // scale of the closest flame (dsBest ~ its world size) drives both
        // the minimum step and the density softness. Smoke instances take
        // the smoke branch below instead (own step law + density).
        float dsBest = max(bestInst.posScale.w *
                           min(max(bestInst.sizeParams.y, 1e-3),
                               max(bestInst.sizeParams.x, 1e-3)), 1e-3);
        float wScale = clamp(dsBest, 1.0, 32.0);
        float dt = minStep * wScale;
        float densV = 0.0;
        vec3 emisV = vec3(0.0);
        bool isSmoke = (bestDef.meta.x == SDF_PRIM_SMOKE);
        if (isSmoke) {
            float loopDurS = max(smokeTuning.timing.y, 1.0);
            float loopTS = smokeLoopT(time, loopDurS);
            vec3 scS = bestInst.posScale.xyz;
            float maxRS = max(bestDef.params0.x, 1.0);
            float seedS = bestInst.rotSeed.w;
            float densMulS = max(bestMat.volumeParams.x, 0.0) * max(smokeTuning.wind.z, 0.0);
            float rNowS = smokeGrowthRadius(maxRS, loopTS, max(smokeTuning.timing.x, 0.5));
            // Smoke step law: exact sphere tracing OUTSIDE the volume;
            // INSIDE, a uniform step tied to the noise wavelength so the
            // layered features are sampled finely enough to avoid marched
            // plane banding across a huge (256 m) ball. Dense smoke
            // terminates early, so the finer steps stay cheap.
            float wl = 1.0 / clamp(smokeTuning.noise.x, 0.001, 0.5);
            float interiorStep = clamp(wl * 0.6, 0.4, max(rNowS * 0.04, 1.0));
            dt = (dBest >= 0.0) ? clamp(dBest * safety, minStep, maxStep) : interiorStep;
            SmokeSample ssm = smokeSampleDensity(p, scS, rNowS, maxRS, smokeTuning, time,
                                                 seedS, densMulS);
            dbgSmokeSDF = ssm.sdf;
            dbgSmokeDens = ssm.density;
            dbgSmokeBullet = ssm.bullet;
            dbgSmokeTunnel = ssm.tunnel;
            dbgSmokePress = ssm.pressure;
            dbgSmokeWave = ssm.wave;
            dbgSmokeTurb = ssm.turb;
            dbgSmokeWake = ssm.wake;
            dbgSmokeFinal = ssm.finalD;
            // Accumulate the FINAL density (post tunnel-thinning, wave and
            // loop-end fade), so the render matches debug view 9 and the
            // bullet visibly carves the cloud instead of ghosting through it.
            densV = ssm.finalD;
            if (densV > 0.001) {
                vec3 sunDirW = -normalize(ubo.lightDirection);
                float ltSm = 1.0;
                emisV = smokeShade(p, rd, scS, rNowS, maxRS, smokeTuning, bestMat, time,
                                   seedS, densMulS,
                                   sunDirW, ubo.lightColor, ltSm);
                dbgSmokeLight = ltSm;
            }
        } else {
        if (dBest < eps) {
            // Volume/emissive modes treat the interior as dense (no hard surface).
            dBest = eps;
        }

        // Fire step law: for deformed definitions (noise-deform bit0 or a
        // FLAME with spikes) the step is additionally capped at the deform
        // feature scale dtCap = dsBest (noise features are O(1) canonical
        // units, i.e. about one instance size). This keeps the deformed
        // bands sampled instead of skipping them with a maxStep-sized jump;
        // undeformed primitives keep the original plain maxStep clamp.
        bool deformStep = ((bestDef.meta.z & 1u) != 0u) || (bestDef.meta.x == SDF_PRIM_FLAME);
        float dtCap = dsBest;
        dt = deformStep
             ? clamp(dBest * safety, minStep * wScale, min(maxStep, dtCap))
             : clamp(dBest * safety, minStep * wScale, maxStep);
        // Interior of a deformed field: sample evenly at half the deform
        // feature scale even when dBest was clamped to eps above (the raw
        // sample said we are inside), instead of taking minStep-sized jumps.
        if (deformStep && insideField) {
            dt = min(dt, dtCap * 0.5);
        }
        float soft = 0.15 * wScale;
        float body = 1.0 - smoothstep(-soft, soft, dBest);
        if (body > 0.001) {
            // Flame-local height fraction drives temperature and the top
            // fade: 0 at the flame base, 1 at its tip. (Container height is
            // useless here — one container spans many flames of all sizes,
            // which painted every flame a single flat temperature.)
            float hn = clamp((p.y - bMin.y) / containerH, 0.0, 1.0);
            if (bestDef.meta.x == SDF_PRIM_FLAME) {
                float bds;
                vec3 bq = sdfWorldToLocal(p, bestInst, bds);
                hn = clamp(bq.y / max(bestDef.params0.y, 1e-3), 0.0, 1.0);
            }
            float nse = sdfNoise(p * 0.7 + vec3(bestInst.rotSeed.w * 19.0));
            float temp = clamp((1.0 - hn) * (0.35 + 0.65 * clamp(body, 0.0, 1.0)) + (nse - 0.5) * 0.35, 0.0, 1.0);
            temp *= max(bestMat.volumeParams.w, 0.0) * max(bestInst.sizeParams.z, 0.0);
            vec4 ve = sdfEvaluateVolume(dBest, hn, nse, bestMat, temp);
            densV = ve.a;
            emisV = ve.rgb;
        } // end if (body > 0.001), fire path
        } // end else (non-smoke path)
        // Shared front-to-back accumulation for fire and smoke volumes.
        if (densV > 0.001) {
            float extinct = clamp(densV * (0.5 + bestMat.volumeParams.y), 0.0, 4.0);
            float aStep = clamp(1.0 - exp(-extinct * dt * 2.0), 0.0, 1.0);
            if (tFirst < 0.0 && (1.0 - trans) + aStep * trans > 0.03) tFirst = t;
            float wgt = trans * aStep;
            accum += wgt * emisV;
            tDepthAccum += wgt * t; // weights telescope to alpha: centroid = sum/alpha
            trans *= (1.0 - aStep);
            if ((1.0 - trans) >= opacityThresh) break; // early termination
        }
        t += dt;
    }

    // Smoke debug views (§24): 0 = normal smoke (falls through), 1-10 below.
    // Stored in debugFlags bits 4-7 so the fire debug views (bits 0-2) are
    // unaffected.
    uint smokeDbg = (debugFlags >> 4u) & 15u;
    if (smokeDbg >= 1u && smokeDbg <= 10u) {
        vec3 sd = vec3(0.0);
        if (smokeDbg == 1u) {
            // Base smoke SDF sign: blue = inside, white = boundary, red = outside.
            float b = max(containerH * 0.02, 1.0);
            if (abs(dbgSmokeSDF) < b) sd = vec3(1.0);
            else if (dbgSmokeSDF < 0.0) sd = mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 0.0, 0.1), clamp(-dbgSmokeSDF / (b * 8.0), 0.0, 1.0));
            else sd = mix(vec3(1.0, 0.0, 0.0), vec3(0.1, 0.0, 0.0), clamp(dbgSmokeSDF / (b * 8.0), 0.0, 1.0));
        } else if (smokeDbg == 2u) {
            sd = vec3(clamp(dbgSmokeDens, 0.0, 1.0));
        } else if (smokeDbg == 3u) {
            float v = clamp(dbgSmokeBullet / 50.0, -1.0, 1.0);
            sd = (v < 0.0) ? vec3(-v, 0.0, 0.0) : vec3(0.0, v, v * 0.5);
        } else if (smokeDbg == 4u) {
            sd = vec3(clamp(dbgSmokeTunnel, 0.0, 1.0));
        } else if (smokeDbg == 5u) {
            sd = vec3(clamp(abs(dbgSmokePress), 0.0, 1.0));
        } else if (smokeDbg == 6u) {
            float w = clamp(dbgSmokeWave * 0.5 + 0.5, 0.0, 1.0);
            sd = vec3(w, w * 0.6, w * 0.3);
        } else if (smokeDbg == 7u) {
            sd = vec3(clamp(dbgSmokeTurb, 0.0, 1.0));
        } else if (smokeDbg == 8u) {
            sd = vec3(clamp(dbgSmokeWake, 0.0, 1.0));
        } else if (smokeDbg == 9u) {
            sd = vec3(clamp(dbgSmokeFinal, 0.0, 1.0));
        } else {
            float h = maxSteps > 0 ? clamp(float(steps) / float(maxSteps), 0.0, 1.0) : 0.0;
            sd = mix(vec3(0.0, 0.1, 0.0), vec3(0.0, 1.0, 0.3), h);
        }
        outColor = vec4(sd, 1.0);
        gl_FragDepth = sdfProjDepth(ro, rd, (tFirst >= 0.0) ? tFirst : tEnter);
        return;
    }

    if (debugView) {
        // Bit 0 enables a debug view; bits 1-2 select which one:
        //   0 = march steps (traversal cost heat)
        //   1 = candidate hits (BLACK = ray never evaluated any flame SDF:
        //       grid empty along the ray or every bound test rejected)
        //   2 = field proximity (WHITE = ray passed within ~0 m of a flame
        //       surface; BLACK = closest approach > 3 m: SDF never got small)
        uint dbgMode = (debugFlags >> 1u) & 3u;
        if (dbgMode == 1u) {
            float h = clamp(float(hits) / 24.0, 0.0, 1.0);
            outColor = vec4(sdfTemperatureColor(h), 1.0);
        } else if (dbgMode == 2u) {
            float c = 1.0 - smoothstep(0.0, 3.0, minAbsD);
            outColor = vec4(sdfTemperatureColor(c), 1.0);
        } else {
            float heat = maxSteps > 0 ? clamp(float(steps) / float(maxSteps) * 1.2,
                                              0.0, 1.0) : 0.0;
            outColor = vec4(sdfTemperatureColor(heat), 1.0);
        }
        gl_FragDepth = sdfProjDepth(ro, rd, (tFirst >= 0.0) ? tFirst : tEnter);
        return;
    }

    if (hit) {
        outColor = vec4(hitColor, 1.0);
        gl_FragDepth = sdfProjDepth(ro, rd, hitT);
        return;
    }
    float alpha = 1.0 - trans;
    if (alpha < 0.01) discard;
    outColor = vec4(accum, alpha);
    // Fire depth = radiance centroid (transmittance-weighted mean distance),
    // NOT first contact: entry wisps would otherwise glue the whole flame to
    // the nearest depth and smear it over occluding terrain. The centroid
    // sits in the visible flame mass, so the composite depth test occludes
    // correctly and nearer flame pixels win over farther ones.
    float tDepth = (tFirst >= 0.0) ? tFirst : tEnter;
    if (alpha > 1e-4) tDepth = tDepthAccum / alpha;
    gl_FragDepth = sdfProjDepth(ro, rd, tDepth);
}

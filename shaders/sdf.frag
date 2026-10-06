#version 450

#include "types/Bullet.glsl"
#include "types/SdfContainer.glsl"
#include "types/SdfDefinition.glsl"
#include "types/SdfDeformFlags.glsl"
#include "types/SdfGridCell.glsl"
#include "types/SdfInstance.glsl"
#include "types/SdfMaterial.glsl"
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
#include "includes/sdf_model.glsl"
#include "includes/sdf_noise.glsl"
#include "includes/sdf_smoke.glsl"
// Shared wind field (set 0, binding 27) for the flame lean below. Requires
// perlin.glsl (perlinNoise3D) before wind_field.glsl. Set 0 is the global
// scene set, already first in this pipeline's layout (SdfRenderer), whose
// binding 27 carries VERTEX|FRAGMENT|COMPUTE stage flags — no C++ change.
#include "includes/perlin.glsl"
#include "includes/wind_field.glsl"

layout(std430, set = 1, binding = 0) readonly buffer SdfInstanceBuffer {
    SdfInstance sdfInstances[];
};
layout(std430, set = 1, binding = 1) readonly buffer SdfDefinitionBuffer {
    SdfDefinition sdfDefinitions[];
};
layout(std430, set = 1, binding = 2) readonly buffer SdfMaterialBuffer {
    SdfMaterial sdfMaterials[];
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
layout(std140, set = 1, binding = 6) uniform SdfParamsBlock {
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

// World -> local through the generic SdfModel (sdf_model.glsl): inverse TRS.
// Returns the local point; outScale carries the conservative local->world
// distance scale so sphere tracing never oversteps.
vec3 sdfWorldToLocal(vec3 wpos, SdfInstance inst, out float outScale) {
    return sdfModelToLocal(sdfModelFromInstance(inst), wpos, outScale);
}

float sdfEvalInstance(vec3 wpos, SdfInstance inst, SdfDefinition def,
                      SdfMaterial mat, float time) {
    // Every primitive is evaluated in the local frame supplied by its
    // SdfModel; the smoke primitive marches inside that frame and never
    // performs any transform itself.
    float ds;
    vec3 q = sdfWorldToLocal(wpos, inst, ds);
    if (def.prim == SDF_PRIM_SMOKE) {
        return smokeMarchSDF(q, def, time) * ds;
    }
    // Optional repeat before primitive eval (SDF_DEFORM_REPEAT): period from
    // params1.xyz.
    uint deform = def.deformFlags;
    if ((deform & SDF_DEFORM_REPEAT) != 0u) {
        q = opRepeat(q, abs(def.params1.xyz));
    }
    float d = sdfPrimitive(q, def.prim, def.params0, def.params1);
    // Deformation runs in canonical flame space (flame ~3.2 units tall) so
    // waviness/spike SIZE stays constant under instance scaling; in raw
    // local units an x32 flame would get x32-stretched blobby features.
    // (Offsets stay in local units, consistent with d.)
    vec3 qn = q;
    float hhn = 3.2;
    if (def.prim == SDF_PRIM_FLAME) {
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
        if (((deform & SDF_DEFORM_NOISE) != 0u) || (spkAmp > 0.001)) {
            float turbResp = 0.25 + clamp(mat.turbulence, 0.0, 2.0);
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
    if ((deform & SDF_DEFORM_NOISE) != 0u) {
        float turb = mat.turbulence;
        float rise = mat.riseSpeed;
        d += sdfFlameDeform(qn, time, inst.seed, turb, rise);
        deformed = true;
    }
    if (def.prim == SDF_PRIM_FLAME) {
        // Tapered-flame spikes (params1.y = amplitude, .z = frequency):
        // ridged tongues over the smooth capsule; 0 = rounded capsule.
        float spk = max(def.params1.y, 0.0);
        if (spk > 0.001) {
            d += sdfFlameSpikes(qn, hhn, inst.seed, def.params1.z, spk);
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

vec3 sdfSurfaceNormal(vec3 p, SdfInstance inst, SdfDefinition def,
                      SdfMaterial mat, float time, float e) {
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
    SdfContainer cont = sdfContainers[fragContainerIndex];
    vec3 bMin = cont.boundsMin;
    vec3 bMax = cont.boundsMax;
    uvec3 dim = uvec3(max(cont.resX, 1u), max(cont.resY, 1u), max(cont.resZ, 1u));
    uint cellBase = cont.cellStart;
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

    // Dithered march start (up to an eighth of a noise wavelength): breaks
    // the coherent per-pixel step phase that drew dashed rings along cloud
    // silhouettes (each ray entered the volume at the same step offset).
    // Scaled by the noise wavelength so small/flame scenes stay unaffected.
    {
        float wlJ = 1.0 / clamp(smokeGpu.tuning.noiseScale, 0.001, 0.5);
        float jU = float(sdfHashU(uvec3(uvec2(gl_FragCoord.xy), 7u))) * (1.0 / 4294967295.0);
        tEnter += jU * 0.125 * wlJ;
    }

    // March params from SdfParamsUBO (canonical std140 block, natural
    // fields). Zero-filled UBO -> safe defaults.
    int maxSteps = int(sdfParams.maxSteps + 0.5);
    if (maxSteps <= 0) maxSteps = 64;
    maxSteps = min(maxSteps, SDF_MAX_STEPS_HARD);
    float safety = sdfParams.safety;
    if (safety <= 0.0) safety = 0.7;
    safety = clamp(safety, 0.1, 1.0);
    float eps = sdfParams.epsilon;
    if (eps <= 0.0) eps = 0.01;
    float opacityThresh = sdfParams.earlyTerm;
    if (opacityThresh <= 0.0) opacityThresh = 0.99;
    opacityThresh = clamp(opacityThresh, 0.01, 1.0);
    float minStep = sdfParams.minStep;
    if (minStep <= 0.0) minStep = 0.05;
    float maxStep = sdfParams.maxStep;
    if (maxStep <= 0.0) maxStep = 1.0;
    maxStep = max(maxStep, minStep);
    float time = sdfParams.time;
    uint renderMode = sdfParams.renderMode; // 0 surface,1 volume,2 emissive,3 transparent
    uint debugFlags = sdfParams.debugFlags;
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
    float dbgSmokeHeat = 0.0;
    float dbgSmokeSampled = 0.0; // 1 once a field sample was collected
    // Peak-density tracker: field debug views (2,4-9,11) report the sample
    // that contributed most, NOT the last sub-loop sample (which sits at
    // the far exit in thin air and would read ~0 for every ray). SDF views
    // (1,3) instead track closest approach (min over samples).
    float dbgPeakD = -1e30;
    bool hit = false;
    float hitT = tEnter;
    vec3 hitColor = vec3(0.0);
    // Two-phase smoke resolve: set once the volume's thickness has been
    // measured and shaded, so later contacts march through without
    // double-counting while fire ahead still accumulates.
    bool smokeResolved = false;

    // Visible tracer round: ONE analytic gold sphere of the bullet's own
    // radius at the head, evaluated ONCE before the march (the old
    // per-sample sphere test aliased into stacked horizontal discs, and the
    // multi-sphere chain shaded every sub-sphere with its own highlight, so
    // it read as several smaller balls). Exact ray/sphere root gives a
    // banding-free hit distance. The smoke container expands over the flight
    // path, so the proxy is covered even outside the smoke ball.
    float bestT = 1e5;
    vec3 bestC = vec3(0.0);
    float bestR = 0.0;
    for (int bi = 0; bi < 8; ++bi) {
        Bullet bbl = smokeGpu.bullets[bi];
        if (bbl.intensity <= 0.0) continue;
        SmokeBulletState bst = smokeBulletState(bbl, time);
        if (!bst.live || !bst.headOnPath) continue;
        vec3 bD = bbl.velocity / max(length(bbl.velocity), 1e-6);
        vec3 head = bbl.start + bD * bst.traveled;
        // One sphere, bullet's own radius (max of both ends, guard floor).
        float radius = max(max(bbl.radiusStart, bbl.radiusEnd), 0.3);
        vec3 oc = ro - head;
        float bq = dot(oc, rd);
        float cq = dot(oc, oc) - radius * radius;
        float disc = bq * bq - cq;
        if (disc > 0.0) {
            float tHit = -bq - sqrt(disc);
            if (tHit > 0.0 && tHit < bestT) {
                bestT = tHit;
                bestC = head;
                bestR = radius;
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
                            * max(smokeGpu.tuning.goldNormalDistort, 0.0);
            vec3 bNt = (dot(bNp, bNp) > 1e-12) ? normalize(bNp) : bN;
            float pat = sdfNoise(hp * max(smokeGpu.tuning.goldPatternScale, 0.0) + bestC);
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
            float specPower = smokeGpu.tuning.goldSpecPower;
            float spec = (specPower > 1e-3) ? pow(specDot, specPower) : 0.0;
            float fres = pow(clamp(1.0 - max(dot(bNt, V), 0.0), 0.0, 1.0), 3.0);
            vec3 gold = mix(smokeGpu.tuning.goldDeep, smokeGpu.tuning.goldBright,
                            clamp(pat * 0.65 + fres * smokeGpu.tuning.goldFresnelBoost, 0.0, 1.0));
            vec3 goldCol = gold * (ubo.lightColor * (0.25 + 0.9 * dif)
                                   + vec3(1.0, 0.72, 0.25) * smokeGpu.tuning.goldWarmFloor)
                         + ubo.lightColor * spec * smokeGpu.tuning.goldSpecStrength;
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
        SdfInstance bestInst;
        SdfDefinition bestDef;
        SdfMaterial bestMat;
        bool haveBest = false;
        for (uint k = 0u; k < SDF_MAX_CANDIDATES; k++) {
            if (k >= n) break;
            uint gAddr = coff + k;
            if (gAddr >= uint(nIndices)) break;
            uint ii = sdfGridIndices[gAddr];
            if (ii >= uint(nInstances)) continue;
            SdfInstance inst = sdfInstances[ii];
            uint di = inst.defIdx;
            uint mi = inst.matIdx;
            if (di >= uint(nDefs) || mi >= uint(nMats)) continue;
            SdfMaterial m0 = sdfMaterials[mi];
            vec3 oq = max(max(inst.boundsMin - p, p - inst.boundsMax), vec3(0.0));
            float ob = length(oq); // exact distance to the instance AABB
            float skipR = maxStep + 0.1 + abs(m0.turbulence);
            if (ob > skipR) {
                nearestBox = min(nearestBox, ob); // conservative advance source
                continue;
            }
            SdfDefinition dd = sdfDefinitions[di];
            float d = sdfEvalInstance(p, inst, dd, m0, time);
            float kk = clamp(sdfUnpackSmoothK(dd), 0.0, 2.0);
            if (!haveBest) { dBest = d; haveBest = true; }
            else { dBest = sdfCombine(dBest, d, dd.op, kk); }
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

        // Surface mode (or any mode hitting the zero crossing): shade opaque
        // hit. Solid smoke shapes (Sphere/Cube) are opaque bodies in every
        // render mode, matching the reference plastic sphere/cube.
        bool smokeSolid = (bestDef.prim == SDF_PRIM_SMOKE) &&
                          (smokeGpu.tuning.shape > 0.5);
        if (dBest < eps && (renderMode == 0u || renderMode == 3u || smokeSolid)) {
            float e = max(eps * 2.0, 0.004);
            vec3 nn = sdfSurfaceNormal(p, bestInst, bestDef, bestMat, time, e);
            vec3 L = -normalize(ubo.lightDirection);
            float ndl = max(dot(nn, L), 0.0);
            float alpha = clamp(bestMat.opacity, 0.0, 1.0);
            hitColor = bestMat.baseColor.rgb * (0.2 + ndl) * ubo.lightColor + bestMat.emission * 0.2;
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
        float dsBest = max(bestInst.scale *
                           min(max(bestInst.radiusScale, 1e-3),
                               max(bestInst.heightScale, 1e-3)), 1e-3);
        float wScale = clamp(dsBest, 1.0, 32.0);
        float dt = minStep * wScale;
        float densV = 0.0;
        vec3 emisV = vec3(0.0);
        bool isSmoke = (bestDef.prim == SDF_PRIM_SMOKE);
        // ---- two-phase smoke resolve (volume/emissive modes) ----
        // Phase A measures the smoke depth analytically (exact ray/sphere
        // roots = what a depth march converges to); phase B shades the
        // measured thickness with a fixed-count uniform march. Constant cost
        // per ray: no step-budget exhaustion, no view-dependent landing, so
        // the look no longer tracks the camera. Surface/transparent modes
        // keep the old zero-crossing hit above and never enter here.
        if (isSmoke && !smokeSolid && !smokeResolved && renderMode != 0u && renderMode != 3u) {
            float loopDurS = max(smokeGpu.tuning.loopDuration, 1.0);
            float loopTS = smokeLoopT(time, loopDurS);
            vec3 scS = bestInst.position;
            float shapeScaleS = max(bestDef.params0.x, 1.0);
            float seedS = bestInst.seed;
            float densMulS = max(bestMat.density, 0.0) * max(smokeGpu.tuning.densityScale, 0.0);
            float rNowS = smokeGrowthRadius(shapeScaleS, loopTS, max(smokeGpu.tuning.growthDuration, 0.5));
            // Smoke samples and the sun/wind directions live in the shape's
            // local frame; the generic SdfModel provides the mapping.
            SdfModel smokeModel = sdfModelFromInstance(bestInst);
            vec3 windLocalS = sdfModelDirToLocal(smokeModel, vec3(smokeGpu.tuning.wind.x, 0.0, smokeGpu.tuning.wind.y));
            vec3 rdLocalS = sdfModelDirToLocal(smokeModel, rd);
            vec3 sunLocalS = sdfModelDirToLocal(smokeModel, -normalize(ubo.lightDirection));
            // Trigger on analytic containment (not the carved SDF), so the
            // bullet bore can never hide the volume from the resolver.
            float contS = length(p - scS) - rNowS;
            if (contS < eps) {
                vec3 ocS = ro - scS;
                float tcaS = -dot(ocS, rd);
                float ddS = dot(ocS, ocS) - tcaS * tcaS;
                float rrS = rNowS * rNowS;
                float tA0 = tEnter, tA1 = tEnter - 1.0;
                if (ddS < rrS) {
                    float thcS = sqrt(max(rrS - ddS, 0.0));
                    tA0 = max(tcaS - thcS, tEnter);
                    tA1 = min(tcaS + thcS, tExit);
                }
                float loS = max(t, tA0);
                if (tA1 > loS) {
                    // Phase B: fixed-count uniform march across the measured
                    // thickness. Jittered start; each sample shaded with its
                    // marched-in depth (deep mass reads darker, stably).
                    float thickS = max(tA1 - loS, 1e-3);
                    float dtS = thickS / 12.0;
                    float jS = float(sdfHashU(uvec3(uvec2(gl_FragCoord.xy), 19u))) * (1.0 / 4294967295.0);
                    float ts = loS + jS * dtS;
                    for (int j = 0; j < 12; ++j) {
                        if (ts > tA1) break;
                        vec3 ps = ro + rd * ts; // world sample along the ray
                        // Local sample in the shape's frame: the evaluator
                        // owns the transform, the smoke module never sees it.
                        float dsS;
                        vec3 qs = sdfWorldToLocal(ps, bestInst, dsS);
                        SmokeSample ssm = smokeSampleDensity(qs, rNowS, shapeScaleS, windLocalS,
                                                             smokeGpu.tuning, time, seedS, densMulS);
                        // Closest-approach views (SDF sign, bullet distance):
                        // min over every sub-sample, ungated.
                        dbgSmokeSDF = min(dbgSmokeSDF, ssm.sdf);
                        dbgSmokeBullet = min(dbgSmokeBullet, ssm.bullet);
                        // Reference field views latch their own peak, ungated
                        // by density: air compression (5), ripple wave (6),
                        // heat (11).
                        dbgSmokePress = max(dbgSmokePress, ssm.pressure);
                        dbgSmokeWave = max(dbgSmokeWave, ssm.wave);
                        dbgSmokeHeat = max(dbgSmokeHeat, ssm.heat);
                        dbgSmokeSampled = 1.0;
                        float densS = ssm.finalD;
                        if (densS > 0.001) {
                            float ltSm = 1.0;
                            vec3 emisS = smokeShade(qs, rdLocalS, rNowS, shapeScaleS, windLocalS,
                                                   smokeGpu.tuning, bestMat, time,
                                                   seedS, densMulS,
                                                   sunLocalS, ubo.lightColor, densS, ssm.heat, ltSm);
                            // Depth shading: brightness falls gently with
                            // marched-in depth, so the lit face reads
                            // against the mass without hollowing the core.
                            float depthInS = max(ts - loS, 0.0);
                            emisS *= 0.55 + 0.45 * exp(-depthInS * 0.002);
                            // Peak-density sample owns the density views; the
                            // compression/ripple/heat fields latch their own
                            // peaks above (reference semantics).
                            if (densS > dbgPeakD) {
                                dbgPeakD = densS;
                                dbgSmokeDens = ssm.density;
                                dbgSmokeTunnel = ssm.tunnel;
                                dbgSmokeTurb = ssm.turb;
                                dbgSmokeWake = ssm.wake;
                                dbgSmokeFinal = ssm.finalD;
                                dbgSmokeLight = ltSm;
                            }
                            float extinctS = clamp(densS * (0.5 + bestMat.absorption) * 0.08, 0.0, 4.0);
                            float aStepS = clamp(1.0 - exp(-extinctS * dtS * 2.0), 0.0, 1.0);
                            if (tFirst < 0.0 && (1.0 - trans) + aStepS * trans > 0.03) tFirst = ts;
                            float wgtS = trans * aStepS;
                            accum += wgtS * emisS;
                            tDepthAccum += wgtS * ts;
                            trans *= (1.0 - aStepS);
                            if ((1.0 - trans) >= opacityThresh) break;
                        }
                        // Hot bore glow: heat emits even where the smoke ran
                        // thin, so the tunnel reads instead of vanishing.
                        // Additive only (never occludes); heat-strength slider
                        // matches the reference uHeatStrength (0 = off).
                        if (ssm.heat > 0.01 && smokeGpu.tuning.heatStrength > 0.0) {
                            vec3 heatWarm = mix(vec3(smokeGpu.tuning.smokeColor),
                                               vec3(1.0, 0.48, 0.15), 0.75);
                            accum += trans * heatWarm
                                   * (ssm.heat * smokeGpu.tuning.heatStrength) * dtS;
                        }
                        ts += dtS;
                    }
                    smokeResolved = true;
                    t = tA1 + 1e-4;
                    if ((1.0 - trans) >= opacityThresh) break;
                    continue;
                }
                // Analytic miss (grazing resolved by the SDF): sphere-trace
                // through without accumulating.
                dt = clamp(dBest * safety, minStep, maxStep);
            } else {
                // Approaching the ball: exact sphere tracing to its surface.
                dt = clamp(dBest * safety, minStep, maxStep);
            }
        } else if (isSmoke) {
            // Volume already resolved behind us (or a surface mode, handled
            // by the zero-crossing hit above): sphere-trace through without
            // accumulating so fire ahead still marches normally.
            dt = clamp(dBest * safety, minStep, maxStep);
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
        bool deformStep = ((bestDef.deformFlags & 1u) != 0u) || (bestDef.prim == SDF_PRIM_FLAME);
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
            if (bestDef.prim == SDF_PRIM_FLAME) {
                float bds;
                vec3 bq = sdfWorldToLocal(p, bestInst, bds);
                hn = clamp(bq.y / max(bestDef.params0.y, 1e-3), 0.0, 1.0);
            }
            float nse = sdfNoise(p * 0.7 + vec3(bestInst.seed * 19.0));
            float temp = clamp((1.0 - hn) * (0.35 + 0.65 * clamp(body, 0.0, 1.0)) + (nse - 0.5) * 0.35, 0.0, 1.0);
            temp *= max(bestMat.tempScale, 0.0) * max(bestInst.intensity, 0.0);
            vec4 ve = sdfEvaluateVolume(dBest, hn, nse, bestMat, temp);
            densV = ve.a;
            emisV = ve.rgb;
        } // end if (body > 0.001), fire path
        } // end else (non-smoke path)
        // Shared front-to-back accumulation for fire and smoke volumes.
        if (densV > 0.001) {
            // Smoke-only extinction scale 0.08: the material values are
            // tuned for dense little flames; a cloud hundreds of meters
            // across needs a per-meter coefficient an order of magnitude
            // smaller or one step goes opaque (which is what made the look
            // track the camera). ~e-fold per 90 m at typical density.
            // Flames keep their original coefficient.
            float extinctScale = isSmoke ? 0.08 : 1.0;
            float extinct = clamp(densV * (0.5 + bestMat.absorption) * extinctScale, 0.0, 4.0);
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
    // Cloud-only: solids shade as opaque bodies and have no volume samples
    // to visualize (a latched view would otherwise paint their background).
    if (smokeGpu.tuning.shape <= 0.5 && smokeDbg >= 1u && smokeDbg <= 11u) {
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
            // Air compression (reference field view): shell/stagnation/
            // rarefaction, same +1/2.5 mapping, peak along the ray. Rays
            // that miss the smoke get the reference's dark field backdrop.
            sd = (dbgSmokeSampled > 0.5) ? smokeDebugRamp((dbgSmokePress + 1.0) / 2.5)
                                         : vec3(0.015, 0.02, 0.04);
        } else if (smokeDbg == 6u) {
            // Compression ripple ("wave"): signed wall ripple mapped 0..1.
            sd = (dbgSmokeSampled > 0.5) ? smokeDebugRamp(0.5 + 0.5 * dbgSmokeWave)
                                         : vec3(0.015, 0.02, 0.04);
        } else if (smokeDbg == 7u) {
            // Signed velocity field magnitude (m/s): white = full
            // entrainment speed (wakeExpansion * bullet speed, ~77 m/s).
            sd = vec3(clamp(dbgSmokeTurb / 80.0, 0.0, 1.0));
        } else if (smokeDbg == 8u) {
            sd = vec3(clamp(dbgSmokeWake, 0.0, 1.0));
        } else if (smokeDbg == 9u) {
            sd = vec3(clamp(dbgSmokeFinal, 0.0, 1.0));
        } else if (smokeDbg == 11u) {
            // Heat (reference field view): wall band + lingering wake trail.
            sd = (dbgSmokeSampled > 0.5) ? smokeDebugRamp(dbgSmokeHeat)
                                         : vec3(0.015, 0.02, 0.04);
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

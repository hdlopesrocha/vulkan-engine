#version 450

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
    float raw = texture(sdfSceneDepth, cuv).r;
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
    }
    if ((deform & 1u) != 0u) {
        float turb = mat.extra.z;
        float rise = mat.extra.w;
        d += sdfFlameDeform(qn, time, inst.rotSeed.w, turb, rise);
    }
    if (def.meta.x == SDF_PRIM_FLAME) {
        // Tapered-flame spikes (params1.y = amplitude, .z = frequency):
        // ridged tongues over the smooth capsule; 0 = rounded capsule.
        float spk = max(def.params1.y, 0.0);
        if (spk > 0.001) {
            d += sdfFlameSpikes(qn, hhn, inst.rotSeed.w, def.params1.z, spk);
        }
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
    uint indexBase = cont.gridInfo.w; // == gridOffset.y (global index start)

    vec3 ro = ubo.viewPosition;
    vec3 toFrag = fragWorldPos - ro;
    float toFragLen = length(toFrag);
    if (toFragLen < 1e-8) discard;
    vec3 rd = toFrag / toFragLen;

    float tEnter, tExit;
    if (!sdfRayAabb(ro, rd, bMin, bMax, tEnter, tExit)) discard;
    tEnter = max(tEnter, 0.0);

    // NOTE: no in-pass scene-depth clamp. The async SDF task only waits on
    // tlCull, so the solid depth target is still an attachment (or stale)
    // here — sampling it raced the solid pass and discarded everything.
    // Occlusion against opaque geometry is resolved in the composite
    // (postprocess.frag), which runs after tlSdf AND tlSolid and compares
    // the volume depth against the scene depth. sdfSceneDepth stays bound
    // but unsampled.
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
    bool hit = false;
    float hitT = tEnter;
    vec3 hitColor = vec3(0.0);

    for (int i = 0; i < SDF_MAX_STEPS_HARD; i++) {
        if (i >= maxSteps || t > tExit) break;
        steps = i + 1;
        vec3 p = ro + rd * t;

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
            coff = sdfGridCells[cellAddr].offset + indexBase;
            ccnt = sdfGridCells[cellAddr].count;
        }
        if (ccnt == 0u) {
            t = t + dtCell + 1e-4; // empty-space skipping: one DDA jump
            continue;
        }

        uint n = min(ccnt, SDF_MAX_CANDIDATES);
        float dBest = 1e5;
        float dMin = 1e5;
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
            float skipR = maxStep + 0.1 + abs(m0.extra.z);
            if (length(oq) > skipR) continue; // conservative bounds reject
            SdfDefinitionGPU dd = sdfDefinitions[di];
            float d = sdfEvalInstance(p, inst, dd, m0, time);
            float kk = clamp(sdfUnpackSmoothK(dd), 0.0, 2.0);
            if (!haveBest) { dBest = d; haveBest = true; }
            else { dBest = sdfCombine(dBest, d, dd.meta.y, kk); }
            if (d < dMin) { dMin = d; bestInst = inst; bestDef = dd; bestMat = m0; }
        }
        if (!haveBest) { t = t + dtCell + 1e-4; continue; }
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
        if (dBest < eps) {
            // Volume/emissive modes treat the interior as dense (no hard surface).
            dBest = eps;
        }

        // Scale-aware volumetric stepping: the SDF distance scale of the
        // closest flame (dsBest ~ its world size) drives both the minimum
        // step and the density softness. Fixed 0.05 m steps could never
        // penetrate a 100 m flame (all 64 steps die in the cool skin, so the
        // hot core never contributes); fixed 0.15 m softness gave razor
        // silhouettes instead of soft volumetric edges. Scaled together the
        // edge is always ~3 steps wide and the march reaches the core.
        float dsBest = max(bestInst.posScale.w *
                           min(max(bestInst.sizeParams.y, 1e-3),
                               max(bestInst.sizeParams.x, 1e-3)), 1e-3);
        float wScale = clamp(dsBest, 1.0, 32.0);
        float dt = clamp(dBest * safety, minStep * wScale, maxStep);
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
            float extinct = clamp(ve.a * (0.5 + bestMat.volumeParams.y), 0.0, 4.0);
            float aStep = clamp(1.0 - exp(-extinct * dt * 2.0), 0.0, 1.0);
            if (tFirst < 0.0 && (1.0 - trans) + aStep * trans > 0.03) tFirst = t;
            float wgt = trans * aStep;
            accum += wgt * ve.rgb;
            tDepthAccum += wgt * t; // weights telescope to alpha: centroid = sum/alpha
            trans *= (1.0 - aStep);
            if ((1.0 - trans) >= opacityThresh) break; // early termination
        }
        t += dt;
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

// Water TCS (moved from water.tesc). Requires: ubo, locations,
// perlin, water_noise, waterParams/waterRenderUBO (ubo.glsl).
// Defines the stage's main().


// Water tessellation control shader — noise-adaptive tessellation
// Uses the same Perlin FBM noise function as the displacement/bump to
// allocate more triangles in areas with higher wave activity.

layout(vertices = 3) out;

layout(location = VARY_LOCALPOS) in vec3 inPos[];
layout(location = VARY_NORMAL) in vec3 inNormal[];
layout(location = VARY_SHARPNORMAL) in vec3 inBaseNormal[];
layout(location = VARY_BRUSHPATCH) flat in int pc_inBrushIndex[];
layout(location = VARY_HSV) in vec3 inHSV[];

layout(location = VARY_LOCALPOS) out vec3 outPos[];
layout(location = VARY_NORMAL) out vec3 outNormal[];
layout(location = VARY_SHARPNORMAL) out vec3 outBaseNormal[];
layout(location = VARY_BRUSHPATCH) flat out ivec3 tc_fragBrushIndex[];
layout(location = VARY_TEXWEIGHTS) out vec3 tc_fragTexWeights[];
layout(location = VARY_HSV) out vec3 tc_fragHSV[];


void main() {
    outPos[gl_InvocationID] = inPos[gl_InvocationID];
    outNormal[gl_InvocationID] = inNormal[gl_InvocationID];
    outBaseNormal[gl_InvocationID] = inBaseNormal[gl_InvocationID];

    int i0 = pc_inBrushIndex[0];
    int i1 = pc_inBrushIndex[1];
    int i2 = pc_inBrushIndex[2];

    int u0 = i0;
    int u1 = (i1 == u0) ? -1 : i1;
    int u2;
    if (i2 == u0 || (u1 != -1 && i2 == u1)) u2 = -1;
    else u2 = i2;

    tc_fragBrushIndex[gl_InvocationID] = ivec3(u0, u1, u2);

    int myIdx = pc_inBrushIndex[gl_InvocationID];
    vec3 texWeights = vec3(0.0);
    if (myIdx == u0) texWeights.x = 1.0;
    else if (myIdx == u1) texWeights.y = 1.0;
    else if (myIdx == u2) texWeights.z = 1.0;

    tc_fragHSV[gl_InvocationID] = inHSV[gl_InvocationID];
    tc_fragTexWeights[gl_InvocationID] = texWeights;

    if (gl_InvocationID == 0) {
        // Respect the global tessellation toggle from settings.
        if (ubo.passParams.y < 0.5) {
            gl_TessLevelOuter[0] = 1.0;
            gl_TessLevelOuter[1] = 1.0;
            gl_TessLevelOuter[2] = 1.0;
            gl_TessLevelInner[0] = 1.0;
            return;
        }

        // Select the water params from the first brush index on the patch.
        // Brush ids are terrain paint ids; the water SSBO holds only a few
        // layers, so out-of-range ids fall back to layer 0 (default look)
        // instead of reading past the allocation.
        int idx = max(pc_inBrushIndex[0], 0);
        int nWL = max(waterParams.length(), 1);
        WaterParamsGPU wp = waterParams[(idx >= 0 && idx < nWL) ? idx : 0];

        float nearDist   = wp.tessParams.x;
        float farDist    = wp.tessParams.y;
        float minLevel   = wp.tessParams.z;
        float maxLevel   = wp.tessParams.w;
        float noiseInf   = wp.waveParams.x;
        float timeVal    = waterRenderUBO.timeParams.x * wp.params3.x;

        // Per-edge tessellation: evaluate noise at each edge's midpoint so
        // adjacent patches sharing an edge compute the same midpoint, the same
        // noise value, and therefore the same outer tess level — no cracks.
        //   e=0: edge(v0,v1) -> opposite v2 -> Outer[2]
        //   e=1: edge(v1,v2) -> opposite v0 -> Outer[0]
        //   e=2: edge(v2,v0) -> opposite v1 -> Outer[1]
        float outer[3];
        for (int e = 0; e < 3; ++e) {
            vec3 va = inPos[e];
            vec3 vb = inPos[(e + 1) % 3];

            // min(da,db) is symmetric: both adjacent patches get the same value
            float da = length(ubo.viewPos.xyz - va);
            float db = length(ubo.viewPos.xyz - vb);
            float distTess = clamp(farDist / max(min(da, db), 1.0), minLevel, maxLevel);

            if (noiseInf > 0.0 && wp.waveToggles.x > 0.5) {
                // Noise at edge midpoint — deterministic, identical for both
                // adjacent patches sharing this edge, so the two can never
                // disagree on the shared tess level (no cracks). The TCS has no
                // thickness signal, so the probe runs as deep water (-1).
                // H7: one ridged train of two octaves instead of the full
                // field — this is a triangle-density bias, not a look.
                vec3 edgeMid = (va + vb) * 0.5;
                float noiseVal = waterWaveTessProbe(edgeMid, timeVal, wp);
                float noiseMod = 1.0 + noiseInf * (noiseVal - 0.5);
                outer[e] = clamp(distTess * noiseMod, minLevel, maxLevel);
            } else {
                outer[e] = distTess;
            }
        }

        gl_TessLevelOuter[0] = outer[1];  // edge(v1,v2), opposite v0
        gl_TessLevelOuter[1] = outer[2];  // edge(v2,v0), opposite v1
        gl_TessLevelOuter[2] = outer[0];  // edge(v0,v1), opposite v2
        gl_TessLevelInner[0] = max(max(outer[0], outer[1]), outer[2]);
    }
}

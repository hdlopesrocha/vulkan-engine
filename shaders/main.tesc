#version 450

layout(vertices = 3) out;

#include "includes/ubo.glsl"
#include "includes/locations.glsl"

// Pass through per-vertex varyings from vertex shader
layout(location = VARY_COLOR) in vec3 pc_inFragColor[];
layout(location = VARY_UV) in vec2 pc_inUV[];
layout(location = VARY_NORMAL) in vec3 pc_inNormal[];
layout(location = VARY_POSWORLD) in vec3 pc_inPosWorld[];
layout(location = VARY_BRUSHPATCH) in int pc_inBrushIndex[];
layout(location = VARY_LOCALPOS) in vec3 pc_inLocalPos[];
layout(location = VARY_LOCALNORMAL) in vec3 pc_inLocalNormal[];


layout(location = VARY_COLOR) out vec3 tc_fragColor[];
layout(location = VARY_UV) out vec2 tc_fragUV[];
layout(location = VARY_NORMAL) out vec3 tc_fragNormal[];
layout(location = VARY_POSWORLD) out vec3 tc_fragPosWorld[];
layout(location = VARY_BRUSHPATCH) flat out ivec3 tc_fragBrushIndex[];
layout(location = VARY_LOCALPOS) out vec3 tc_fragLocalPos[];
layout(location = VARY_LOCALNORMAL) out vec3 tc_fragLocalNormal[];
layout(location = VARY_TEXWEIGHTS) out vec3 tc_fragTexWeights[];


// Compute tessellation factor for a single edge.  All inputs come from the
// edge's own two vertex positions and brush indices, guaranteeing that two
// adjacent patches visiting their shared edge always compute the same outer
// tessellation level — the necessary condition for crack-free LOD transitions.
float computeEdgeTess(vec3 a, vec3 b, int brushA, int brushB) {
    // Deterministic edge material: pick the lower non-negative brush index.
    // min() is symmetric, so both adjacent patches — which may list the
    // shared edge's vertices in opposite order — select the same material.
    int edgeBrush;
    if (brushA >= 0 && brushB >= 0)  edgeBrush = min(brushA, brushB);
    else if (brushA >= 0)            edgeBrush = brushA;
    else if (brushB >= 0)            edgeBrush = brushB;
    else                             edgeBrush = 0;

    float nearDist = ubo.tessParams.x;
    float farDist  = ubo.tessParams.y;
    float factor_g = ubo.tessParams.z;

    // Per-edge tessellation enable: if globally off or this edge's material
    // does not request tessellation, return 1.0.  Both adjacent patches
    // compute the same edgeBrush so they reach the same decision.
    if (ubo.passParams.y < 0.5 || materials[edgeBrush].mappingParams.x < 0.5)
        return 1.0;

    float minLevel = materials[edgeBrush].tessLevelParams.x * factor_g;
    float maxLevel = materials[edgeBrush].tessLevelParams.y * factor_g;
    float matBoost = materials[edgeBrush].mappingParams.y;

    float da = length(a - ubo.viewPos.xyz);
    float db = length(b - ubo.viewPos.xyz);
    float d  = min(da, db);
    float t  = clamp(1.0 - smoothstep(nearDist, farDist, d), 0.0, 1.0);
    return mix(minLevel, maxLevel, t) + matBoost * t;
}

void main() {
    // Pass through per-vertex data to evaluation stage
    tc_fragColor[gl_InvocationID] = pc_inFragColor[gl_InvocationID];
    tc_fragUV[gl_InvocationID] = pc_inUV[gl_InvocationID];
    tc_fragNormal[gl_InvocationID] = pc_inNormal[gl_InvocationID];
    tc_fragPosWorld[gl_InvocationID] = pc_inPosWorld[gl_InvocationID];


    // Compress the patch's texture indices into up to three unique slots.
    // Discard -1 (underground) brush indices so they never contribute texture.
    int i0 = pc_inBrushIndex[0];
    int i1 = pc_inBrushIndex[1];
    int i2 = pc_inBrushIndex[2];

    int u0 = (i0 >= 0) ? i0 : -1;
    int u1 = -1;
    int u2 = -1;
    if (i1 >= 0 && i1 != u0) u1 = i1;
    if (i2 >= 0 && i2 != u0 && i2 != u1) u2 = i2;

    // Store the unique indices (use -1 for empty slots)
    tc_fragBrushIndex[gl_InvocationID] = ivec3(u0, u1, u2);

    // Map this corner's barycentric basis into the matching unique-slot.
    // Vertices with brushIndex < 0 (underground) get zero weight so they
    // never bleed the fallback material (index 0 = bricks) into the blend.
    int myIdx = pc_inBrushIndex[gl_InvocationID];
    vec3 texWeights = vec3(0.0);
    if (myIdx >= 0) {
        if (myIdx == u0) texWeights.x = 1.0;
        else if (myIdx == u1) texWeights.y = 1.0;
        else if (myIdx == u2) texWeights.z = 1.0;
    }

    tc_fragTexWeights[gl_InvocationID] = texWeights;
    tc_fragLocalPos[gl_InvocationID] = pc_inLocalPos[gl_InvocationID];
    tc_fragLocalNormal[gl_InvocationID] = pc_inLocalNormal[gl_InvocationID];
    // tangents are computed in the fragment shader for triplanar mapping

    // Per-edge tessellation: all material lookups and distance computations
    // use only the edge's own two vertex positions and brush indices.  Both
    // adjacent patches sharing an edge see the same vertex pair, so they
    // always produce identical outer levels — no LOD crack seams.
    //   Outer0 = edge opposite v0 = edge (v1, v2)
    //   Outer1 = edge opposite v1 = edge (v2, v0)
    //   Outer2 = edge opposite v2 = edge (v0, v1)
    vec3 p0 = pc_inPosWorld[0];
    vec3 p1 = pc_inPosWorld[1];
    vec3 p2 = pc_inPosWorld[2];
    float outer0 = computeEdgeTess(p1, p2, pc_inBrushIndex[1], pc_inBrushIndex[2]);
    float outer1 = computeEdgeTess(p2, p0, pc_inBrushIndex[2], pc_inBrushIndex[0]);
    float outer2 = computeEdgeTess(p0, p1, pc_inBrushIndex[0], pc_inBrushIndex[1]);
    float inner  = max(max(outer0, outer1), outer2);

    gl_TessLevelOuter[0] = outer0;
    gl_TessLevelOuter[1] = outer1;
    gl_TessLevelOuter[2] = outer2;
    gl_TessLevelInner[0] = inner;
}
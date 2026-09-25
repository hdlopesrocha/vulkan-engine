// EVSM shadow sampling with gap-free cascade selection.
// Cascade index is determined by light-space projection (matching GPU
// cascade culling).  Cascade 2 (outermost) is sampled directly without
// bounds check so fragments near any cascade boundary always get a
// valid shadow value instead of falling through to return 0.0.
#include "evsm.glsl"

bool insideShadowMap(vec3 p, float margin) {
    return p.x >= margin && p.x <= 1.0 - margin &&
           p.y >= margin && p.y <= 1.0 - margin &&
           p.z >= 0.0    && p.z <= 1.0;
}

vec3 projectToShadowMap(mat4 lightSpaceMat, vec3 worldPos) {
    vec4 lsPos = lightSpaceMat * vec4(worldPos, 1.0);
    vec3 p = lsPos.xyz / lsPos.w;
    p.xy = p.xy * 0.5 + 0.5;
    return p;
}

float cascadeBlendFactor(vec2 uv, float margin) {
    vec2 edgeDist = min(uv, 1.0 - uv);
    float minEdge = min(edgeDist.x, edgeDist.y);
    return clamp(minEdge / margin, 0.0, 1.0);
}

float cascadeBlendZ(float z, float margin) {
    float distFromFar = 1.0 - z;
    return clamp(distFromFar / margin, 0.0, 1.0);
}

// Shared blend margin (was a body-local const).
const float SHADOW_BLEND_MARGIN = 0.04;

// Forward declaration: the 4-arg core is defined below; the wrapper needs it
// in scope (GLSL free functions must be declared before use).
float ShadowCalculation(vec4 fragPosLightSpace, vec3 worldPos, float bias, out int cascadeHint);

float ShadowCalculation(vec4 fragPosLightSpace, vec3 worldPos, float bias) {
    int hint;
    return ShadowCalculation(fragPosLightSpace, worldPos, bias, hint);
}

// Cascade selection + sampling with the selected cascade reported (perf
// report 21 H6). Bit-identical values to the historical behavior; the hint
// seeds the secondary-hit fast path below at negligible primary cost.
float ShadowCalculation(vec4 fragPosLightSpace, vec3 worldPos, float bias, out int cascadeHint) {
    // Global shadow toggle. The raster solid/vegetation/impostor paths gate
    // their own calls, but the RT reflection/refraction shading (inline ray
    // hits and the mirror/bounce chains) samples the CSM unconditionally.
    // Without this guard a disabled CSM still darkens every secondary hit
    // through its stale or zero-initialized cascade maps — the "shadow
    // underwater" that survives turning shadows off.
    // Hint default: cascade 0 (also covers the disabled-shadows early-out,
    // whose hint would otherwise be undefined). Overwritten below wherever
    // the selection lands in cascade 1 or 2.
    cascadeHint = 0;
    if (!ubo.shadowsEnabled) return 0.0;

    const float BLEND_MARGIN = 0.04;

    vec3 proj0 = fragPosLightSpace.xyz / fragPosLightSpace.w;
    proj0.xy = proj0.xy * 0.5 + 0.5;

    // Cascade 0 — try with extended bounds for blending.
    // Blend based on BOTH XY proximity to the shadow-map edge AND Z
    // proximity to the far plane (cascade depth boundary).  This
    // smooths the transition between cascades and eliminates visible
    // seams at the depth split.
    if (insideShadowMap(proj0, -BLEND_MARGIN)) {
        float s0 = ShadowEVSM(shadowMap, proj0, bias);
        float blendXY = cascadeBlendFactor(proj0.xy, BLEND_MARGIN);
        float blendZ  = cascadeBlendZ(proj0.z, BLEND_MARGIN);
        float blend0  = min(blendXY, blendZ);
        if (blend0 >= 1.0) return s0;

        // Blend with cascade 1
        vec3 proj1 = projectToShadowMap(ubo.lightSpaceMatrix1, worldPos);
        float s1 = insideShadowMap(proj1, 0.0)
            ? ShadowEVSM(shadowMap1, proj1, bias) : s0;
        return mix(s1, s0, smoothstep(0.0, 1.0, blend0));
    }

    // Cascade 1
    vec3 proj1 = projectToShadowMap(ubo.lightSpaceMatrix1, worldPos);
    if (insideShadowMap(proj1, -BLEND_MARGIN)) {
        float s1 = ShadowEVSM(shadowMap1, proj1, bias);
        float blendXY = cascadeBlendFactor(proj1.xy, BLEND_MARGIN);
        float blendZ  = cascadeBlendZ(proj1.z, BLEND_MARGIN);
        float blend1  = min(blendXY, blendZ);
        if (blend1 >= 1.0) { cascadeHint = 1; return s1; }

        // Blend with cascade 2
        vec3 proj2 = projectToShadowMap(ubo.lightSpaceMatrix2, worldPos);
        float s2 = insideShadowMap(proj2, 0.0)
            ? ShadowEVSM(shadowMap2, proj2, bias) : s1;
        cascadeHint = 1;
        return mix(s2, s1, smoothstep(0.0, 1.0, blend1));
    }

    // Cascade 2 — outermost cascade: sample directly without bounds
    // check.  This guarantees every fragment receives a valid shadow
    // value even near cascade boundaries where light-space projections
    // may fall just outside all cascade AABBs.
    vec3 proj2 = projectToShadowMap(ubo.lightSpaceMatrix2, worldPos);
    cascadeHint = 2;
    return ShadowEVSM(shadowMap2, proj2, bias);
}

float ShadowCalculationHard(vec4 fragPosLightSpace, vec3 worldPos, float bias) {
    return ShadowCalculation(fragPosLightSpace, worldPos, bias);
}

// Strict-inside test (perf report 21 H6): the full path returns its
// single-cascade early-out exactly when the blend factor reaches 1.0, i.e.
// XY at least one blend margin inside and Z short of the far plane by the
// same margin. A strict-inside hit therefore resolves to the identical
// single sample — the fast path below is never approximate. Anything else
// (edge zones, behind-light projections) falls back to the full blend path.
bool insideShadowMapStrict(vec3 p) {
    return p.x >= SHADOW_BLEND_MARGIN && p.x <= 1.0 - SHADOW_BLEND_MARGIN &&
           p.y >= SHADOW_BLEND_MARGIN && p.y <= 1.0 - SHADOW_BLEND_MARGIN &&
           p.z >= 0.0                 && p.z <= 1.0 - SHADOW_BLEND_MARGIN;
}

// Secondary-hit shadow with a primary-cascade hint (H6). EXACT, not
// approximate: the full path always tests cascade 0 first, so this
// reproduces that test (one matvec, as today) and only past it lets the hint
// skip the remaining selection — project the hinted cascade directly and
// take the strict-inside single sample, which the blend math proves is the
// value the full path would return (blend factor 1.0). Edge hits, wrong
// hints and hint < 1 (unknown, cascade 0, water entries) fall back to the
// single full-path call below with the projection reused, i.e. textually
// today's call. Savings land on strict cascade-1/2 secondaries: up to one
// matvec, the blend-factor ALU and the blend-zone second fetch.
float ShadowCalculationSecondary(vec3 hitPos, int hintCascade, float bias) {
    vec4 hitProj0 = ubo.lightSpaceMatrix * vec4(hitPos, 1.0);
    vec3 p0 = hitProj0.xyz / hitProj0.w;
    p0.xy = p0.xy * 0.5 + 0.5;
    // Fast lane needs cascade 0 missed AND a usable hint; the hint
    // projection is evaluated only then (pure ALU, safe under divergence).
    bool useHint = !insideShadowMap(p0, -SHADOW_BLEND_MARGIN)
                && (hintCascade == 1 || hintCascade == 2);
    vec3 hp = useHint
        ? projectToShadowMap(hintCascade == 1 ? ubo.lightSpaceMatrix1 : ubo.lightSpaceMatrix2, hitPos)
        : vec3(0.0);
    if (useHint && insideShadowMapStrict(hp)) {
        if (hintCascade == 1) return ShadowEVSM(shadowMap1, hp, bias);
        return ShadowEVSM(shadowMap2, hp, bias);
    }
    int unusedHint;
    return ShadowCalculation(hitProj0, hitPos, bias, unusedHint);
}

// EVSM shadow sampling replaces the old 5x5 PCF and hard-shadow functions.
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

// Compute blend factor [0,1] for cascade transition: 1 = well inside, 0 = at/outside edge
float cascadeBlendFactor(vec2 uv, float margin) {
    vec2 edgeDist = min(uv, 1.0 - uv);
    float minEdge = min(edgeDist.x, edgeDist.y);
    return clamp(minEdge / margin, 0.0, 1.0);
}

float ShadowCalculation(vec4 fragPosLightSpace, vec3 worldPos, float bias) {
    const float BLEND_MARGIN = 0.04;

    vec3 proj0 = fragPosLightSpace.xyz / fragPosLightSpace.w;
    proj0.xy = proj0.xy * 0.5 + 0.5;

    // Cascade 0 — try with extended bounds for blending
    if (insideShadowMap(proj0, -BLEND_MARGIN)) {
        float s0 = ShadowEVSM(shadowMap, proj0, bias);
        float blend0 = cascadeBlendFactor(proj0.xy, BLEND_MARGIN);
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
        float blend1 = cascadeBlendFactor(proj1.xy, BLEND_MARGIN);
        if (blend1 >= 1.0) return s1;

        // Blend with cascade 2
        vec3 proj2 = projectToShadowMap(ubo.lightSpaceMatrix2, worldPos);
        float s2 = insideShadowMap(proj2, 0.0)
            ? ShadowEVSM(shadowMap2, proj2, bias) : s1;
        return mix(s2, s1, smoothstep(0.0, 1.0, blend1));
    }

    // Cascade 2
    vec3 proj2 = projectToShadowMap(ubo.lightSpaceMatrix2, worldPos);
    if (insideShadowMap(proj2, 0.0)) {
        return ShadowEVSM(shadowMap2, proj2, bias);
    }

    return 0.0;
}

float ShadowCalculationHard(vec4 fragPosLightSpace, vec3 worldPos, float bias) {
    return ShadowCalculation(fragPosLightSpace, worldPos, bias);
}

// EVSM shadow sampling replaces the old 5x5 PCF and hard-shadow functions.
#include "evsm.glsl"

// Returns true when projCoords.xy is within [margin, 1-margin] and z in [0,1].
bool insideShadowMap(vec3 projCoords, float margin) {
    return projCoords.x >= margin && projCoords.x <= 1.0 - margin &&
           projCoords.y >= margin && projCoords.y <= 1.0 - margin &&
           projCoords.z >= 0.0  && projCoords.z <= 1.0;
}

// Project a world-space position into a light-space shadow map and return
// normalized [0,1] coordinates (xy) and depth (z).
vec3 projectToShadowMap(mat4 lightSpaceMat, vec3 worldPos) {
    vec4 lsPos = lightSpaceMat * vec4(worldPos, 1.0);
    vec3 p = lsPos.xyz / lsPos.w;
    p.xy = p.xy * 0.5 + 0.5;
    return p;
}

// Blend weight at cascade edge: 0 = well inside, 1 = at exact edge
float edgeBlendWeight(vec2 uv, float margin) {
    vec2 d = min(uv, 1.0 - uv);
    return 1.0 - clamp(min(d.x, d.y) / margin, 0.0, 1.0);
}

// Extended bounds check for bidirectional blending — accepts UV slightly
// outside [0,1] so the neighbour cascade can sample back across the border.
bool nearShadowMapEdge(vec3 projCoords, float margin) {
    return projCoords.x >= -margin && projCoords.x <= 1.0 + margin &&
           projCoords.y >= -margin && projCoords.y <= 1.0 + margin &&
           projCoords.z >= 0.0     && projCoords.z <= 1.0;
}

const float CASCADE_MARGIN = 0.08;

float ShadowCalculation(vec4 fragPosLightSpace, vec3 worldPos, float bias) {
    // Cascade 0
    vec3 proj0 = fragPosLightSpace.xyz / fragPosLightSpace.w;
    proj0.xy = proj0.xy * 0.5 + 0.5;

    float shadow = 0.0;

    if (insideShadowMap(proj0, 0.0)) {
        shadow = ShadowEVSM(shadowMap, proj0, bias);
        // Blend into cascade 1 when near edge.
        // Use (1-blend) so that at the boundary (blend=1) shadow = shadow0,
        // matching the reverse-blend below.
        float blend = edgeBlendWeight(proj0.xy, CASCADE_MARGIN);
        if (blend > 0.0) {
            vec3 proj1 = projectToShadowMap(ubo.lightSpaceMatrix1, worldPos);
            if (insideShadowMap(proj1, 0.0)) {
                float s1 = ShadowEVSM(shadowMap1, proj1, bias);
                shadow = mix(s1, shadow, 1.0 - blend);
            }
        }
        return shadow;
    }

    // Also project into cascade 0 for bidirectional blending below
    // (when already outside but near the boundary).
    bool nearBorder0 = nearShadowMapEdge(proj0, CASCADE_MARGIN);

    // Cascade 1
    vec3 proj1 = projectToShadowMap(ubo.lightSpaceMatrix1, worldPos);
    if (insideShadowMap(proj1, 0.0)) {
        shadow = ShadowEVSM(shadowMap1, proj1, bias);
        // Blend back into cascade 0 when near cascade-0's boundary.
        // Use proj0's distance to the [0,1] edge (via clamp) as the blend weight.
        if (nearBorder0) {
            vec3 p0 = clamp(proj0, 0.0, 1.0);
            float blend = edgeBlendWeight(p0.xy, CASCADE_MARGIN);
            float s0 = ShadowEVSM(shadowMap, p0, bias);
            shadow = mix(shadow, s0, blend);
        }
        return shadow;
    }

    // Cascade 2
    vec3 proj2 = projectToShadowMap(ubo.lightSpaceMatrix2, worldPos);
    if (insideShadowMap(proj2, 0.0)) {
        shadow = ShadowEVSM(shadowMap2, proj2, bias);
        // Blend back into cascade 1 when near cascade-1's boundary
        if (nearShadowMapEdge(proj1, CASCADE_MARGIN)) {
            vec3 p1 = clamp(proj1, 0.0, 1.0);
            float blend = edgeBlendWeight(p1.xy, CASCADE_MARGIN);
            float s1 = ShadowEVSM(shadowMap1, p1, bias);
            shadow = mix(shadow, s1, blend);
        }
        return shadow;
    }

    return 0.0;
}

float ShadowCalculationHard(vec4 fragPosLightSpace, vec3 worldPos, float bias) {
    return ShadowCalculation(fragPosLightSpace, worldPos, bias);
}

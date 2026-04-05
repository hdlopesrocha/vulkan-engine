// Shadow PCF helper: samples a single shadow map with a 5x5 kernel.
float ShadowPCF(sampler2D smap, vec3 projCoords, float bias) {
    float currentDepth = projCoords.z;
    float shadow = 0.0;
    vec2 texelSize = 1.0 / textureSize(smap, 0);
    for (int x = -2; x <= 2; ++x) {
        for (int y = -2; y <= 2; ++y) {
            float pcfDepth = texture(smap, projCoords.xy + vec2(x, y) * texelSize).r;
            shadow += currentDepth > (pcfDepth + bias) ? 1.0 : 0.0;
        }
    }
    return shadow / 25.0;
}

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

// Cascaded Shadow Map calculation:
//   Try cascade 0 (finest) first; if the fragment is outside its UV range,
//   fall back to cascade 1, then cascade 2.
float ShadowCalculation(vec4 fragPosLightSpace, float bias) {
    // Cascade 0: use the interpolated light-space position from vertex/tese shader
    vec3 proj0 = fragPosLightSpace.xyz / fragPosLightSpace.w;
    proj0.xy = proj0.xy * 0.5 + 0.5;

    if (insideShadowMap(proj0, 0.0)) {
        return ShadowPCF(shadowMap, proj0, bias);
    }

    // Cascade 1
    vec3 proj1 = projectToShadowMap(ubo.lightSpaceMatrix1, fragPosWorld);
    if (insideShadowMap(proj1, 0.0)) {
        return ShadowPCF(shadowMap1, proj1, bias);
    }

    // Cascade 2
    vec3 proj2 = projectToShadowMap(ubo.lightSpaceMatrix2, fragPosWorld);
    if (insideShadowMap(proj2, 0.0)) {
        return ShadowPCF(shadowMap2, proj2, bias);
    }

    // Outside all cascades: no shadow
    return 0.0;
}
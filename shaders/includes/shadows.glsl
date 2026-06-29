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

float ShadowCalculation(vec4 fragPosLightSpace, vec3 worldPos, float bias) {
    vec3 proj0 = fragPosLightSpace.xyz / fragPosLightSpace.w;
    proj0.xy = proj0.xy * 0.5 + 0.5;

    if (insideShadowMap(proj0, 0.0))
        return ShadowEVSM(shadowMap, proj0, bias);

    vec3 proj1 = projectToShadowMap(ubo.lightSpaceMatrix1, worldPos);
    if (insideShadowMap(proj1, 0.0))
        return ShadowEVSM(shadowMap1, proj1, bias);

    vec3 proj2 = projectToShadowMap(ubo.lightSpaceMatrix2, worldPos);
    if (insideShadowMap(proj2, 0.0))
        return ShadowEVSM(shadowMap2, proj2, bias);

    return 0.0;
}

float ShadowCalculationHard(vec4 fragPosLightSpace, vec3 worldPos, float bias) {
    return ShadowCalculation(fragPosLightSpace, worldPos, bias);
}

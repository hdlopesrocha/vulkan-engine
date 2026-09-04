// Shared closest-hit shading helpers: raster-equivalent material evaluation.
// Mirrors shaders/main.frag (albedo/normal/roughness/AO sampling, lighting,
// specular, AO, HSV) with includes/triplanar.glsl + includes/common.glsl +
// includes/tbn.glsl formulas. The environment reflection term keeps the exact
// raster mix factor (k = strength * Schlick) but traces the real scene
// instead of sampling the cubemap. Documented deviation: no dFdx/dFdy in ray
// tracing, so the non-triplanar normal-map TBN uses a stable up-cross basis.
#ifndef RT_SHADING_GLSL
#define RT_SHADING_GLSL

#include "rt_material.glsl"

#include "rt_material.glsl"

// Shadow-ray payload (location 1 pairs with rt_shadow.rmiss). Declared here so
// every closest-hit shader including this file can issue shadow traces.
layout(location = 1) rayPayloadEXT ShadowPayload rtShadowPayload;

float rtTraceShadow(vec3 origin, vec3 lightDir, float lightDist) {
    // Default occluded: with TERMINATE_ON_FIRST_HIT + SKIP_CLOSEST_HIT no hit
    // shader runs, so a blocking intersection keeps 1.0 and only the shadow
    // miss (escaped → lit) resets to 0.0. Vegetation any-hit discards
    // below-threshold texels so traversal continues past cutout foliage.
    rtShadowPayload.occlusion = 1.0;
    uint flags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT; // no Opaque: any-hit alpha-test must run
    uint mask = uint(rt.traceMask.x + 0.5);
    traceRayEXT(tlas, flags, mask, 0, 0, 1,
                origin, RAY_TMIN, lightDir, lightDist, 1);
    return rtShadowPayload.occlusion;
}

// Clamp a corner brush id to the materials range (or 0 when none exist).
ivec3 rtClampBrush(ivec3 brush) {
    int n = int(materials.length());
    return ivec3(n > 0 ? clamp(brush.x, 0, n - 1) : 0,
                 n > 0 ? clamp(brush.y, 0, n - 1) : 0,
                 n > 0 ? clamp(brush.z, 0, n - 1) : 0);
}

// Full raster-equivalent solid evaluation (mirrors main.frag). worldPos is the
// UNDISPLACED hit point (== fragPosWorldNotDisplaced); smoothN is the
// face-forwarded interpolated normal; geomN the unflipped face normal.
void rtEvalSolid(ivec3 brushIn, vec3 bary, vec2 uv, vec3 worldPos,
                 vec3 smoothN, vec3 geomN,
                 out vec3 albedo, out vec3 worldNormal,
                 out float roughnessValue, out float ambientOcclusion,
                 out float ambientFactor, out float specStrength, out float shininess,
                 out float reflectionStrength, out float roughnessFactor,
                 out float aoFactor, out float useAO, out float aoBlend,
                 out int dominantMat) {
    ivec3 ids = rtClampBrush(brushIn);
    dominantMat = (bary.x >= bary.y && bary.x >= bary.z) ? ids.x : ((bary.y >= bary.z) ? ids.y : ids.z);

    // Triplanar weights from the geometric normal (+ per-corner UV sets).
    vec3 triW = rtTriplanarWeights(geomN);
    float triFlag = materials[ids.x].triplanarParams.z * bary.x
                  + materials[ids.y].triplanarParams.z * bary.y
                  + materials[ids.z].triplanarParams.z * bary.z;
    vec2 uv0X, uv0Y, uv0Z, uv1X, uv1Y, uv1Z, uv2X, uv2Y, uv2Z;
    if (bary.x > 0.0) rtTriplanarUVs(worldPos, ids.x, geomN, uv0X, uv0Y, uv0Z);
    if (bary.y > 0.0) rtTriplanarUVs(worldPos, ids.y, geomN, uv1X, uv1Y, uv1Z);
    if (bary.z > 0.0) rtTriplanarUVs(worldPos, ids.z, geomN, uv2X, uv2Y, uv2Z);

    bool usedTriplanar = triFlag > 0.5;
    float mapBlend = materials[ids.x].mappingParams.x * bary.x
                   + materials[ids.y].mappingParams.x * bary.y
                   + materials[ids.z].mappingParams.x * bary.z;
    bool normalMapping = rt.featureToggles.x > 0.5;
    bool roughnessOn = rt.featureToggles.y > 0.5;
    bool aoOn = rt.featureToggles.z > 0.5;

    if (usedTriplanar) {
        vec3 a0 = bary.x > 0.0 ? (triW.x > 0.0 || triW.y > 0.0 || triW.z > 0.0
                    ? (texture(albedoArray, vec3(uv0X, float(ids.x))).rgb * triW.x
                     + texture(albedoArray, vec3(uv0Y, float(ids.x))).rgb * triW.y
                     + texture(albedoArray, vec3(uv0Z, float(ids.x))).rgb * triW.z) : vec3(0.0)) : vec3(0.0);
        vec3 a1 = bary.y > 0.0 ? (texture(albedoArray, vec3(uv1X, float(ids.y))).rgb * triW.x
                     + texture(albedoArray, vec3(uv1Y, float(ids.y))).rgb * triW.y
                     + texture(albedoArray, vec3(uv1Z, float(ids.y))).rgb * triW.z) : vec3(0.0);
        vec3 a2 = bary.z > 0.0 ? (texture(albedoArray, vec3(uv2X, float(ids.z))).rgb * triW.x
                     + texture(albedoArray, vec3(uv2Y, float(ids.z))).rgb * triW.y
                     + texture(albedoArray, vec3(uv2Z, float(ids.z))).rgb * triW.z) : vec3(0.0);
        albedo = a0 * bary.x + a1 * bary.y + a2 * bary.z;
        if (mapBlend > 0.5 || normalMapping) {
            vec3 t0 = bary.x > 0.0 ? rtTriplanarNormalUVs(triW, ids.x, smoothN, uv0X, uv0Y, uv0Z) : vec3(0.0);
            vec3 t1 = bary.y > 0.0 ? rtTriplanarNormalUVs(triW, ids.y, smoothN, uv1X, uv1Y, uv1Z) : vec3(0.0);
            vec3 t2 = bary.z > 0.0 ? rtTriplanarNormalUVs(triW, ids.z, smoothN, uv2X, uv2Y, uv2Z) : vec3(0.0);
            worldNormal = normalize(t0 * bary.x + t1 * bary.y + t2 * bary.z + vec3(1e-12));
        } else {
            worldNormal = smoothN;
        }
        float r0 = bary.x > 0.0 ? (texture(roughnessArray, vec3(uv0X, float(ids.x))).r * triW.x
                     + texture(roughnessArray, vec3(uv0Y, float(ids.x))).r * triW.y
                     + texture(roughnessArray, vec3(uv0Z, float(ids.x))).r * triW.z) : 0.0;
        float r1 = bary.y > 0.0 ? (texture(roughnessArray, vec3(uv1X, float(ids.y))).r * triW.x
                     + texture(roughnessArray, vec3(uv1Y, float(ids.y))).r * triW.y
                     + texture(roughnessArray, vec3(uv1Z, float(ids.y))).r * triW.z) : 0.0;
        float r2 = bary.z > 0.0 ? (texture(roughnessArray, vec3(uv2X, float(ids.z))).r * triW.x
                     + texture(roughnessArray, vec3(uv2Y, float(ids.z))).r * triW.y
                     + texture(roughnessArray, vec3(uv2Z, float(ids.z))).r * triW.z) : 0.0;
        roughnessValue = clamp(r0 * bary.x + r1 * bary.y + r2 * bary.z, 0.0, 1.0);
        float o0 = bary.x > 0.0 ? (texture(aoArray, vec3(uv0X, float(ids.x))).r * triW.x
                     + texture(aoArray, vec3(uv0Y, float(ids.x))).r * triW.y
                     + texture(aoArray, vec3(uv0Z, float(ids.x))).r * triW.z) : 0.0;
        float o1 = bary.y > 0.0 ? (texture(aoArray, vec3(uv1X, float(ids.y))).r * triW.x
                     + texture(aoArray, vec3(uv1Y, float(ids.y))).r * triW.y
                     + texture(aoArray, vec3(uv1Z, float(ids.y))).r * triW.z) : 0.0;
        float o2 = bary.z > 0.0 ? (texture(aoArray, vec3(uv2X, float(ids.z))).r * triW.x
                     + texture(aoArray, vec3(uv2Y, float(ids.z))).r * triW.y
                     + texture(aoArray, vec3(uv2Z, float(ids.z))).r * triW.z) : 0.0;
        ambientOcclusion = clamp(o0 * bary.x + o1 * bary.y + o2 * bary.z, 0.0, 1.0);
    } else {
        vec3 a0 = texture(albedoArray, vec3(uv, float(ids.x))).rgb;
        vec3 a1 = texture(albedoArray, vec3(uv, float(ids.y))).rgb;
        vec3 a2 = texture(albedoArray, vec3(uv, float(ids.z))).rgb;
        albedo = a0 * bary.x + a1 * bary.y + a2 * bary.z;
        if (mapBlend > 0.5 || normalMapping) {
            vec3 n0 = texture(normalArray, vec3(uv, float(ids.x))).rgb * 2.0 - 1.0;
            vec3 n1 = texture(normalArray, vec3(uv, float(ids.y))).rgb * 2.0 - 1.0;
            vec3 n2 = texture(normalArray, vec3(uv, float(ids.z))).rgb * 2.0 - 1.0;
            n0 = rtApplyNormalConvention(n0, materials[ids.x].normalParams);
            n1 = rtApplyNormalConvention(n1, materials[ids.y].normalParams);
            n2 = rtApplyNormalConvention(n2, materials[ids.z].normalParams);
            vec3 nmap = normalize(n0 * bary.x + n1 * bary.y + n2 * bary.z + vec3(1e-12));
            // No derivatives in ray tracing: stable up-cross basis instead of
            // the rasterizer's dFdx TBN (documented deviation).
            vec3 up = abs(smoothN.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
            vec3 T = normalize(cross(up, smoothN) + vec3(1e-12));
            vec3 B = cross(smoothN, T);
            worldNormal = normalize(T * nmap.x + B * nmap.y + smoothN * nmap.z);
        } else {
            worldNormal = smoothN;
        }
        float r0 = texture(roughnessArray, vec3(uv, float(ids.x))).r;
        float r1 = texture(roughnessArray, vec3(uv, float(ids.y))).r;
        float r2 = texture(roughnessArray, vec3(uv, float(ids.z))).r;
        roughnessValue = clamp(r0 * bary.x + r1 * bary.y + r2 * bary.z, 0.0, 1.0);
        float o0 = texture(aoArray, vec3(uv, float(ids.x))).r;
        float o1 = texture(aoArray, vec3(uv, float(ids.y))).r;
        float o2 = texture(aoArray, vec3(uv, float(ids.z))).r;
        ambientOcclusion = clamp(o0 * bary.x + o1 * bary.y + o2 * bary.z, 0.0, 1.0);
    }
    if (!roughnessOn) roughnessValue = 0.0;

    ambientFactor = materials[ids.x].materialFlags.z * bary.x
                  + materials[ids.y].materialFlags.z * bary.y
                  + materials[ids.z].materialFlags.z * bary.z;
    specStrength = materials[ids.x].specularParams.x * bary.x
                 + materials[ids.y].specularParams.x * bary.y
                 + materials[ids.z].specularParams.x * bary.z;
    shininess = max(materials[ids.x].specularParams.y * bary.x
                  + materials[ids.y].specularParams.y * bary.y
                  + materials[ids.z].specularParams.y * bary.z, 1.0);
    reflectionStrength = clamp(materials[ids.x].tessLevelParams.z * bary.x
                             + materials[ids.y].tessLevelParams.z * bary.y
                             + materials[ids.z].tessLevelParams.z * bary.z, 0.0, 1.0);
    roughnessFactor = materials[ids.x].roughnessAOParams.x * bary.x
                    + materials[ids.y].roughnessAOParams.x * bary.y
                    + materials[ids.z].roughnessAOParams.x * bary.z;
    aoFactor = materials[ids.x].roughnessAOParams.y * bary.x
             + materials[ids.y].roughnessAOParams.y * bary.y
             + materials[ids.z].roughnessAOParams.y * bary.z;
    useAO = materials[ids.x].roughnessAOParams.z * bary.x
          + materials[ids.y].roughnessAOParams.z * bary.y
          + materials[ids.z].roughnessAOParams.z * bary.z;
    float aoB = (useAO > 0.5 && aoOn) ? ambientOcclusion : 1.0;
    aoBlend = mix(1.0, aoB, aoFactor);
}

// Raster-equivalent direct lighting (ambient + diffuse + specular).
// shadowOrigin must already be lifted clear of the traversed base mesh
// (see rtSurfaceOrigin): a plain epsilon from a displaced point can start
// inside the mesh and self-shadow the bump pattern it displaced from.
vec3 rtLightSolid(vec3 albedo, vec3 N, vec3 V, vec3 shadowOrigin,
                  float roughnessValue, float ambientFactor,
                  float specStrength, float shininess, float roughnessFactor,
                  float aoBlend, out float shadowOut) {
    vec3 L = normalize(rt.lightDir.xyz);
    float NdotL = max(dot(N, L), 0.0);
    float occl = 0.0;
    if (rt.rtParams.w > 0.5) {
        if (NdotL > 0.01) {
            occl = rtTraceShadow(shadowOrigin, L, 1e27);
        } else {
            occl = 1.0;
        }
    }
    shadowOut = 1.0 - occl;
    vec3 ambient = albedo * ambientFactor * aoBlend;
    vec3 diffuse = albedo * rt.lightColor.rgb * NdotL * (1.0 - occl);
    vec3 viewDir = V;
    vec3 reflectDir = reflect(-L, N);
    float specPower = max(mix(shininess, 1.0, roughnessValue * roughnessFactor), 1.0);
    float spec = (NdotL > 0.0) ? pow(max(dot(viewDir, reflectDir), 0.0), specPower) : 0.0;
    vec3 specular = rt.lightColor.rgb * spec * (1.0 - occl) * specStrength;
    return ambient + diffuse + specular;
}

#endif // RT_SHADING_GLSL

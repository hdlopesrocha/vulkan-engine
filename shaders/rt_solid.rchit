// Solid closest-hit: raster-equivalent shading (triplanar/UV albedo, normal
// maps, height displacement, AO, HSV) with direct lighting, ray-traced
// shadows, and one bounded reflection bounce using the exact raster Fresnel
// mix factor with a traced scene instead of the cubemap.
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#include "rt_common.glsl"
#include "rt_shading.glsl"

layout(location = 0) rayPayloadInEXT RadiancePayload payload;
hitAttributeEXT vec2 bary;
// NOTE (glslang codegen): traceRayEXT's payload argument MUST be the
// module-scope `payload` (see rt_water.rchit). Nested rays use payload
// FORWARDING with locals for accumulation — never a function-local payload.

void main() {
    uint ci = gl_InstanceCustomIndexEXT;
    uint slot = ci & SLOT_MASK;
    vec3 p0, p1, p2, n0, n1, n2;
    vec2 uv0, uv1, uv2;
    ivec3 brush;
    vec3 h0, h1, h2;
    rtFetchTriangle(false, slot, gl_PrimitiveID, p0, p1, p2, n0, n1, n2, uv0, uv1, uv2, brush, h0, h1, h2);

    vec3 bary3 = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);
    vec3 basePos = p0 * bary3.x + p1 * bary3.y + p2 * bary3.z;
    vec3 geoN = normalize(cross(p1 - p0, p2 - p0) + vec3(1e-12));
    vec3 smoothN = normalize(n0 * bary3.x + n1 * bary3.y + n2 * bary3.z + vec3(1e-12));
    // Double-sided lighting: flip the shading normal on back-face hits so
    // thin/open geometry stays lit instead of black.
    vec3 N = (dot(smoothN, gl_WorldRayDirectionEXT) > 0.0) ? -smoothN : smoothN;
    vec2 uv = uv0 * bary3.x + uv1 * bary3.y + uv2 * bary3.z;

    // Raster-equivalent material evaluation (triplanar == fragPosWorldNotDisplaced).
    vec3 albedo, worldNormal;
    float roughnessValue, ambientOcclusion, ambientFactor, specStrength, shininess;
    float reflectionStrength, roughnessFactor, aoFactor, useAO, aoBlend;
    int domMat;
    rtEvalSolid(brush, bary3, uv, basePos, N, geoN, albedo, worldNormal,
                roughnessValue, ambientOcclusion, ambientFactor, specStrength, shininess,
                reflectionStrength, roughnessFactor, aoFactor, useAO, aoBlend, domMat);

    // TES displacement approximation along the smooth normal (gated by the
    // same global toggle as the rasterizer: flat when tessellation is off).
    float disp;
    vec3 worldPos = rtDisplace(basePos, N, uv, brush, bary3, geoN, disp);
    // All secondary origins lift clear of the traversed base mesh, including
    // negative (inward) displacement that would otherwise start rays inside it.
    vec3 surfOrigin = rtSurfaceOrigin(worldPos, worldNormal, disp);

    vec3 V = normalize(-gl_WorldRayDirectionEXT);
    float shadow;
    vec3 litColor = rtLightSolid(albedo, worldNormal, V, surfOrigin, roughnessValue,
                                 ambientFactor, specStrength, shininess,
                                 roughnessFactor, aoBlend, shadow);

    // Environment reflection with the exact raster mix factor, but traced.
    vec3 R = reflect(-V, worldNormal);
    float cosTheta = clamp(dot(normalize(worldNormal), V), 0.0, 1.0);
    float fresnel = 0.04 + 0.96 * pow(1.0 - cosTheta, 5.0);
    float k = clamp(reflectionStrength * fresnel, 0.0, 1.0);
    float rough = clamp(roughnessValue * roughnessFactor, 0.0, 1.0);
    float maxDepth = rt.rtParams.z;
    int myDepth = payload.bounceCount + 1;
    payload.bounceCount = myDepth;
    bool canBounce = myDepth <= int(maxDepth + 0.5);
    uint radMask = uint(rt.traceMask.x + 0.5);
    vec3 refl = vec3(0.0);
    if (k > 1e-4) {
        if (canBounce) {
            // Payload forwarding: the nested hit reads bounceCount (already
            // myDepth, set at our entry) for its budget, then overwrites it;
            // restore it for our own write-back below.
            traceRayEXT(tlas, gl_RayFlagsNoneEXT, radMask, 0, 0, 0, surfOrigin, RAY_TMIN, R, 1e32, 0);
            refl = payload.radiance;
            payload.bounceCount = myDepth;
        } else {
            refl = rtSkyColor(R);
        }
        refl *= aoBlend * (1.0 - rough * 0.5);
    }
    vec3 col = mix(litColor, refl, k);

    // Per-vertex HSV tint (mirrors main.frag tail).
    vec3 hsvC = h0 * bary3.x + h1 * bary3.y + h2 * bary3.z;
    vec3 texHSV = rtRgbToHsv(col);
    texHSV.x = mod(texHSV.x + hsvC.x, 360.0);
    texHSV.y = clamp(texHSV.y * (hsvC.y * 2.0), 0.0, 1.0);
    texHSV.z *= hsvC.z * 2.0;
    col = rtHsvToRgb(texHSV);

    payload.radiance = col;
    payload.hitT = gl_HitTEXT;
    payload.normal = worldNormal;
    payload.fresnel = fresnel;
    payload.materialId = domMat;
    payload.slotId = int(slot);
    payload.primitiveId = gl_PrimitiveID;
    payload.waterThickness = 0.0;
    payload.bounceCount = myDepth;
    payload.reflectWeight = k;
    payload.refractWeight = 0.0;
    payload.shadow = shadow;
    payload.worldPos = worldPos;
}

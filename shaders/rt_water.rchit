// Water closest-hit: first-class optical material. Fresnel reflection traced
// above the surface through the real TLAS, Snell refraction into the volume,
// ray-derived water thickness (exit - entry) driving Beer-Lambert absorption,
// and depth-dependent shallow/deep scattering. Handles air->water and
// water->air (exiting) with total internal reflection.
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#include "rt_common.glsl"
#include "rt_shading.glsl"

layout(location = 0) rayPayloadInEXT RadiancePayload payload;
hitAttributeEXT vec2 bary;

// NOTE (glslang codegen): traceRayEXT's payload argument MUST be a
// module-scope rayPayloadEXT variable. Function-local payloads are silently
// miscompiled to the incoming payload (verified in SPIR-V disassembly), so
// nested rays use payload FORWARDING: save locals, reset
// payload.bounceCount to this hit's depth, trace into `payload`, read the
// nested result immediately, and write back this hit's final fields at the
// end (including restoring bounceCount + primary identity for the caller).
void traceRadianceMasked(vec3 origin, vec3 dir, uint mask) {
    traceRayEXT(tlas, gl_RayFlagsNoneEXT, mask, 0, 0, 0, origin, RAY_TMIN, dir, 1e32, 0);
}

void main() {
    uint ci = gl_InstanceCustomIndexEXT;
    uint slot = ci & SLOT_MASK;
    vec3 p0, p1, p2, n0, n1, n2;
    vec2 uv0, uv1, uv2;
    ivec3 brush;
    vec3 hw0, hw1, hw2; // per-vertex HSV (unused: water is waterParams-driven)
    rtFetchTriangle(true, slot, gl_PrimitiveID, p0, p1, p2, n0, n1, n2, uv0, uv1, uv2, brush, hw0, hw1, hw2);

    vec3 bary3 = vec3(1.0 - bary.x - bary.y, bary.x, bary.y);
    vec3 worldPos = p0 * bary3.x + p1 * bary3.y + p2 * bary3.z;
    vec3 N = normalize(n0 * bary3.x + n1 * bary3.y + n2 * bary3.z + vec3(1e-12));
    vec3 I = normalize(gl_WorldRayDirectionEXT);
    vec3 V = -I;

    // Water params indexed by brush (same rule as the raster water shader).
    int nwp = int(waterParams.length());
    int wpi = nwp > 0 ? clamp(brush.x, 0, nwp - 1) : -1;

    // Entering (air->water, hit front face) vs exiting (water->air).
    bool entering = dot(I, N) < 0.0;
    vec3 Nf = entering ? N : -N; // face-forward normal (against the ray)

    float ior = rt.waterMisc.x > 0.0 ? rt.waterMisc.x : 1.333;
    float eta = entering ? (1.0 / ior) : ior;
    float f0 = pow((1.0 - ior) / (1.0 + ior), 2.0); // ~0.02 for water
    float cosT = max(dot(-I, Nf), 0.0);
    float F = rtSchlick(cosT, f0);
    float fresnelPower = (wpi >= 0) ? waterParams[wpi].params1.y : 1.0;
    F = pow(F, clamp(fresnelPower, 0.2, 4.0));
    float reflStrength = (wpi >= 0) ? waterParams[wpi].params1.w : 0.6;
    reflStrength *= (rt.waterMisc.y > 0.0 ? rt.waterMisc.y : 1.0);

    float maxDepth = rt.rtParams.z;
    // This hit's depth (rgen starts at 0). Nested rays reset the payload to
    // myDepth before tracing so every secondary gets the same remaining budget.
    int myDepth = payload.bounceCount + 1;
    payload.bounceCount = myDepth;
    bool canBounce = myDepth <= int(maxDepth + 0.5);
    uint radMask = uint(rt.traceMask.x + 0.5);

    // ── Reflection above the surface (real TLAS traversal) ──
    // Self-intersection is avoided with a centimeter-scale origin offset; at
    // grazing angles the ray may still re-hit the same sheet on a deeper
    // bounce, which the budget guard below turns into a sky fallback (never
    // black) instead of another trace.
    vec3 R = reflect(I, Nf);
    vec3 reflCol = vec3(0.0);
    float reflW = 0.0;
    if (F * reflStrength > 0.005) {
        if (canBounce) {
            vec3 ro = worldPos + Nf * RAY_ORIGIN_EPS * 10.0;
            payload.bounceCount = myDepth;
            traceRadianceMasked(ro, R, radMask);
            reflCol = payload.radiance;
            reflW = F * reflStrength;
        } else {
            reflCol = rtSkyColor(R);
            reflW = F * reflStrength;
        }
    }

    // ── Refraction through the volume ──
    vec3 refrDir = refract(I, Nf, eta);
    bool tir = dot(refrDir, refrDir) < 1e-8; // total internal reflection
    vec3 behind = vec3(0.0);
    float thickness = 0.0;
    float transmitW = 0.0;
    if (tir) {
        reflW = max(reflW, reflStrength); // all energy stays above/bounces
        if (reflW <= 0.0) {
            if (canBounce) {
                vec3 ro = worldPos + Nf * RAY_ORIGIN_EPS * 10.0;
                payload.bounceCount = myDepth;
                traceRadianceMasked(ro, R, radMask);
                reflCol = payload.radiance;
            } else {
                reflCol = rtSkyColor(R);
            }
            reflW = 1.0;
        }
    } else if (canBounce) {
        // Transmission ray: solids only (mask 0x01) so the entry surface never
        // self-hits; the first solid behind the water gives both the refracted
        // color (normally shaded) and the ray-derived water thickness.
        vec3 to = worldPos - Nf * RAY_ORIGIN_EPS * 10.0;
        payload.bounceCount = myDepth;
        traceRadianceMasked(to, refrDir, 0x01);
        if (payload.hitT > 0.0) {
            thickness = payload.hitT;
            behind = payload.radiance;
        } else {
            // No terrain behind (grazing ray through open water): deep column.
            thickness = 60.0;
            behind = rtSkyColor(refrDir);
        }
        transmitW = (1.0 - F);
    } else {
        // Bounce budget exhausted: analytic sky behind a thin sheet.
        behind = rtSkyColor(refrDir);
        thickness = 0.0;
        transmitW = (1.0 - F);
    }

    // ── Absorption + depth-dependent body color ──
    vec3 sigma = rt.waterAbsorption.rgb;
    vec3 transmit = rtBeerLambert(sigma, thickness);
    vec3 shallow = (wpi >= 0) ? waterParams[wpi].shallowColor.rgb : vec3(0.15, 0.45, 0.45);
    vec3 deep = (wpi >= 0) ? waterParams[wpi].deepColor.rgb : vec3(0.01, 0.08, 0.16);
    float trans = (wpi >= 0) ? max(waterParams[wpi].shallowColor.a, 0.5) : 4.0;
    float depthMix = 1.0 - exp(-thickness / max(trans, 0.5));
    vec3 body = mix(shallow, deep, clamp(depthMix, 0.0, 1.0));

    // Sun glitter on the surface (matches the raster specular model).
    vec3 L = normalize(rt.lightDir.xyz);
    float specPower = (wpi >= 0) ? max(waterParams[wpi].params3.w, 8.0) : 90.0;
    float specInt = (wpi >= 0) ? waterParams[wpi].params3.z : 0.6;
    float glitter = pow(max(dot(reflect(-L, Nf), V), 0.0), specPower) * specInt;

    vec3 refrCol = behind * transmit * transmitW + body * (1.0 - transmit) * transmitW;
    vec3 col = reflCol * reflW + refrCol + rt.lightColor.rgb * glitter * (1.0 - reflW * 0.5);

    payload.radiance = col;
    payload.hitT = gl_HitTEXT;
    payload.normal = Nf;
    payload.fresnel = F;
    payload.materialId = brush.x;
    payload.slotId = int(slot);
    payload.primitiveId = gl_PrimitiveID;
    payload.waterThickness = thickness;
    payload.bounceCount = myDepth;
    payload.reflectWeight = reflW;
    payload.refractWeight = transmitW;
    payload.shadow = 1.0;
    payload.worldPos = worldPos;
}

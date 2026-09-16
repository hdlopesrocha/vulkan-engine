// Hybrid RT water pipeline: closest-hit shader (proxy triangle boxes).
// Shading is intentionally simple: the proxy is a stable coarse
// representation for SECONDARY visibility (§6/§21), not the displaced/
// tessellated raster surface. Per-box average albedo + analytic face normal
// with a basic sun diffuse gives plausible reflected/refracted colors that
// blend with Fresnel/Beer-Lambert in water.frag.

#version 460
#extension GL_EXT_ray_tracing : require

#include "includes/rt_params.glsl"

layout(set = 0, binding = 3) uniform RTBlock { RayTracingParamsGLSL rt; };
layout(set = 0, binding = 4) readonly buffer ProxyMeta { RTProxyMetaGLSL metas[]; };
layout(set = 0, binding = 6) uniform sampler2D skyEquirectTex;

layout(location = 0) rayPayloadInEXT RTPayload rtPayload;
hitAttributeEXT vec3 bary;

void main() {
    uint boxIdx = rtBoxIndex(uint(gl_PrimitiveID), gl_InstanceCustomIndexEXT);
    RTProxyMetaGLSL meta = metas[boxIdx];
    vec3 hitPos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;
    bool exiting = (gl_HitKindEXT == gl_HitKindBackFacingTriangleEXT);
    vec3 N = rtBoxNormal(hitPos, meta.minAndMatId.xyz, meta.maxAndFlags.xyz, exiting);
    vec3 sunDir = normalize(rt.sunDir.xyz);
    float ndl = max(dot(N, sunDir), 0.0);
    vec3 albedo = meta.albedoRough.rgb;
    // Water proxies (flags=1): the flat tint alone is near-black — a water
    // surface mostly reflects the SKY (Fresnel grows toward grazing), so
    // blend it in for visibility (mirrors main.frag/water.frag).
    vec3 dir = normalize(gl_WorldRayDirectionEXT);
    if (meta.maxAndFlags.w > 0.5) {
        // Own-body guard: a box containing the ray origin is the fragment's
        // own lake surface (the ray starts inside it) — a flat water surface
        // reflects the sky, not itself.
        vec3 o = gl_WorldRayOriginEXT;
        bool ownBody = (o.x >= meta.minAndMatId.x && o.x <= meta.maxAndFlags.x &&
                        o.z >= meta.minAndMatId.z && o.z <= meta.maxAndFlags.z);
        vec3 skyR = texture(skyEquirectTex, rtDirToEquirectUV(dir)).rgb;
        if (ownBody) {
            albedo = skyR;
        } else {
            float fres = rtSchlickFresnel(clamp(dot(N, -dir), 0.0, 1.0), 0.02);
            float wSky = clamp(fres * 1.5 + 0.5, 0.0, 1.0);
            albedo = mix(albedo, skyR, wSky);
        }
    }
    // Ambient + sun diffuse, plus a fixed sky-ambient fill so upward-facing
    // underwater surfaces keep plausible brightness instead of crushing to
    // black under Beer-Lambert (proxy albedo has no sky light otherwise).
    // Shadows stay CSM-owned (§2/§21): RT hits do not recompute the macro
    // sun-shadow solution.
    vec3 color = albedo * (rt.sunColor.rgb * (0.55 + 0.45 * ndl) + vec3(0.09, 0.12, 0.15));
    // Coarse boxes (huge flat tops in the far field) cannot resolve shallow
    // detail. Report the raw hit plus a feather factor; rgen blends toward
    // deep/sky smoothly so box-size contours never print as razor lines.
    rtPayload.coarseF = rtCoarseFeather(meta.extra.x, rt.water.z);
    rtPayload.data = vec4(color, gl_HitTEXT);
}

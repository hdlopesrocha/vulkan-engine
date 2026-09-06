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

layout(location = 0) rayPayloadInEXT vec4 rtPayload;
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
    // Ambient + sun diffuse. Shadows stay CSM-owned (§2/§21): RT hits do not
    // recompute the macro sun-shadow solution.
    vec3 color = albedo * (rt.sunColor.rgb * (0.35 + 0.65 * ndl));
    rtPayload = vec4(color, gl_HitTEXT);
}

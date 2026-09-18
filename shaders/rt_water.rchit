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
    // Constant UP normal for water lighting: the box-face normal varies
    // between the wall's side/top faces per-hit → ndl flickers.
    float ndl = (meta.maxAndFlags.w > 0.5)
        ? max(dot(vec3(0.0, 1.0, 0.0), sunDir), 0.0)
        : max(dot(N, sunDir), 0.0);
    vec3 tint = meta.albedoRough.rgb;
    // Water proxies (flags=1): blend the SKY into the flat tint (Fresnel grows
    // toward grazing). The sky part is a SURFACE mirror and must NOT be
    // multiplied by the sun term — sunlighting the whole mix turned proxies
    // near-black in shadow/grazing, the dark band along the shoreline.
    vec3 dir = normalize(gl_WorldRayDirectionEXT);
    vec3 skyR = vec3(0.0);
    float skyMix = 0.0;
    if (meta.maxAndFlags.w > 0.5) {
        // Own-body guard: a box containing the ray origin is the fragment's
        // own lake surface (the ray starts inside it) — a flat water surface
        // reflects the sky, not itself.
        vec3 o = gl_WorldRayOriginEXT;
        bool ownBody = (o.x >= meta.minAndMatId.x && o.x <= meta.maxAndFlags.x &&
                        o.z >= meta.minAndMatId.z && o.z <= meta.maxAndFlags.z);
        skyR = texture(skyEquirectTex, rtDirToEquirectUV(dir)).rgb;
        if (ownBody) {
            skyMix = 1.0;
        } else {
            // Transparent water look: the water's own tint dominates, with a
            // hint of sky. No recursive reflection/refraction.
            skyMix = 0.2;
            // Soft top edge: rays that graze the wall's top alternate between
            // hitting the wall (water) and passing over it (sky) as the
            // camera moves — a hard switch that flickers. Fade toward the sky
            // near the top so the boundary is continuous.
            float topFade = smoothstep(
                meta.maxAndFlags.y - 15.0, meta.maxAndFlags.y, hitPos.y);
            skyMix = mix(skyMix, 1.0, topFade);
        }
    }
    // Ambient + sun diffuse, plus a sky-ambient fill scaled like the raster's
    // (albedo * ambient) so reflections read as lit scenery, not dark plates.
    // Shadows stay CSM-owned (§2/§21): RT hits do not recompute the macro
    // sun-shadow solution.
    vec3 litTint = tint * (rt.sunColor.rgb * (0.55 + 0.45 * ndl) + vec3(0.26));
    vec3 color = (meta.maxAndFlags.w > 0.5)
        ? mix(litTint, skyR, clamp(skyMix, 0.0, 1.0))
        : litTint;
    // Coarse boxes (huge flat tops in the far field) cannot resolve shallow
    // detail. Report the raw hit plus a feather factor; rgen blends toward
    // deep/sky smoothly so box-size contours never print as razor lines.
    rtPayload.coarseF = rtCoarseFeather(meta.extra.x, rt.water.z);
    rtPayload.data = vec4(color, gl_HitTEXT);
}

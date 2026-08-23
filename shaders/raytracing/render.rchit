#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_tracing_position_fetch : enable
#include "rt_common.glsl"

layout(location = 0) rayPayloadInEXT Payload payload;
layout(location = 1) rayPayloadEXT Payload innerPayload; // outgoing payload for secondary rays

// Shade a hit surface with direct + ambient lighting (no secondary rays). Used
// for reflection/refraction secondary hits so they return a real colour.
vec3 shadeSurface(uint kind, vec3 N, vec3 V, vec3 pos) {
    vec3 albedo = vec3(0.7);
    if (kind == 1u)      albedo = vec3(0.20, 0.42, 0.72); // water
    else if (kind == 2u) albedo = vec3(0.25, 0.60, 0.22); // vegetation
    float ndl = max(dot(N, normalize(-ubo.lightDir.xyz)), 0.0);
    vec3 direct = albedo * (0.15 + 0.85 * ndl) * ubo.lightColor.rgb;
    vec3 ambient = albedo * 0.18 * mix(sky.skyHorizon.rgb, sky.skyZenith.rgb, 0.5);
    return direct + ambient;
}

void main() {
    uint kind = gl_InstanceCustomIndexEXT & 0xFFu;

    // World-space face normal from the hit triangle.
    vec3 p0 = gl_HitTriangleVertexPositionsEXT[0];
    vec3 p1 = gl_HitTriangleVertexPositionsEXT[1];
    vec3 p2 = gl_HitTriangleVertexPositionsEXT[2];
    vec3 N = normalize(cross(p1 - p0, p2 - p0));
    vec3 V = -gl_WorldRayDirectionEXT;
    if (dot(N, V) < 0.0) N = -N;
    vec3 pos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    // ── Shadow ray hit: report occlusion. Vegetation alpha-tested in the
    //    any-hit shader (render.rahit), so any veg hit reaching here is opaque. ──
    if (payload.rayType == 1u) {
        payload.hit = 1.0;
        return;
    }

    // ── Reflection / refraction secondary hit: return shaded surface colour ──
    if (payload.rayType == 2u || payload.rayType == 3u) {
        payload.color = shadeSurface(kind, N, V, pos);
        return;
    }

    // ── Primary hit: full shading + shadow + reflection + refraction ──
    vec3 albedo = vec3(0.7);
    if (kind == 1u)      albedo = vec3(0.20, 0.42, 0.72);
    else if (kind == 2u) albedo = vec3(0.25, 0.60, 0.22);

    vec3 L = normalize(-ubo.lightDir.xyz);
    float ndl = max(dot(N, L), 0.0);

    // Shadow ray toward the light.
    innerPayload.rayType = 1u;
    innerPayload.hit = 0.0;
    innerPayload.color = vec3(0.0);
    innerPayload.normal = vec3(0.0);
    innerPayload.pos = vec3(0.0);
    // No TerminateOnFirstHit: vegetation's any-hit alpha test may reject a
    // transparent leaf (ignoreIntersectionEXT), and the ray must then continue
    // to the opaque geometry behind it rather than terminating as a miss.
    traceRayEXT(tlas, gl_RayFlagsNoneEXT, 0xFFu, 0u, 0u, 0u, pos + N * 0.01, 0.01, L, 10000.0, 1);
    float shadow = innerPayload.hit; // 1.0 = occluded

    vec3 direct = albedo * (0.15 + 0.85 * ndl * (1.0 - shadow)) * ubo.lightColor.rgb;
    vec3 ambient = albedo * 0.18 * mix(sky.skyHorizon.rgb, sky.skyZenith.rgb, 0.5);
    vec3 color = direct + ambient;

    // Reflection (single bounce).
    vec3 reflDir = reflect(gl_WorldRayDirectionEXT, N);
    innerPayload.rayType = 2u;
    innerPayload.hit = 0.0;
    innerPayload.color = vec3(0.0);
    innerPayload.normal = vec3(0.0);
    innerPayload.pos = vec3(0.0);
    traceRayEXT(tlas, gl_RayFlagsNoneEXT, 0xFFu, 0u, 0u, 0u, pos + N * 0.01, 0.01, reflDir, 10000.0, 1);
    float fresnel = 0.04 + 0.96 * pow(1.0 - max(dot(N, V), 0.0), 5.0);
    color = mix(color, innerPayload.color, fresnel);

    // Refraction / transmission (single bounce, glass/water tint).
    vec3 transDir = refract(gl_WorldRayDirectionEXT, N, 0.85);
    if (dot(transDir, transDir) < 1e-4) transDir = gl_WorldRayDirectionEXT; // TIR fallback
    innerPayload.rayType = 3u;
    innerPayload.hit = 0.0;
    innerPayload.color = vec3(0.0);
    innerPayload.normal = vec3(0.0);
    innerPayload.pos = vec3(0.0);
    traceRayEXT(tlas, gl_RayFlagsNoneEXT, 0xFFu, 0u, 0u, 0u, pos - N * 0.01, 0.01, transDir, 10000.0, 1);
    vec3 tint = vec3(0.30, 0.55, 0.80);
    color = mix(color, innerPayload.color * tint, 0.25);

    payload.color = color;
}

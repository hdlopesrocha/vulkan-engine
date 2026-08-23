#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_ray_tracing_position_fetch : enable
#include "rt_common.glsl"

layout(location = 0) rayPayloadInEXT Payload payload;
layout(location = 1) rayPayloadEXT Payload innerPayload; // outgoing payload for reflection/refraction secondary rays

void main() {
    // Geometry/material kind comes from the TLAS instance custom index, so the
    // same unified TLAS + hit group can drive different behavior per geometry
    // type without separate acceleration structures or passes.
    uint kind = gl_InstanceCustomIndexEXT & 0xFFu;

    if (pc.uMode == 0u) {
        // Shadow ray: an opaque triangle hit means the receiver is occluded.
        if (kind == RT_KIND_VEG) {
            // Alpha-test the leaf texture: transparent texels let the shadow ray
            // pass through (no occlusion). UV is reconstructed from the fixed
            // cross-quad topology (6 planes × 2 tris; corner UVs BL=(0,1) BR=(1,1)
            // TL=(0,0) TR=(1,0)) and the hit barycentrics. The foliage array
            // layer rides in the instance custom index high byte.
            uint layer = (gl_InstanceCustomIndexEXT >> 8) & 0xFFu;
            int tri = gl_PrimitiveID % 2;
            vec2 uvA, uvB, uvC;
            if (tri == 0) { uvA = vec2(0.0, 1.0); uvB = vec2(1.0, 1.0); uvC = vec2(0.0, 0.0); }
            else          { uvA = vec2(1.0, 1.0); uvB = vec2(1.0, 0.0); uvC = vec2(0.0, 0.0); }
            vec3 p  = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
            vec3 v0 = gl_HitTriangleVertexPositionsEXT[0];
            vec3 v1 = gl_HitTriangleVertexPositionsEXT[1];
            vec3 v2 = gl_HitTriangleVertexPositionsEXT[2];
            vec3 wvec = p - v0;
            vec3 perp1 = cross(gl_WorldRayDirectionEXT, v2 - v0);
            float d1 = dot(v1 - v0, perp1);
            float u = (abs(d1) > 1e-8) ? dot(wvec, perp1) / d1 : 0.0;
            vec3 perp2 = cross(gl_WorldRayDirectionEXT, v1 - v0);
            float d2 = dot(v2 - v0, perp2);
            float v = (abs(d2) > 1e-8) ? dot(wvec, perp2) / d2 : 0.0;
            vec2 uv = (1.0 - u - v) * uvA + u * uvB + v * uvC;
            float a = texture(rtOpacity, vec3(uv, float(layer))).a;
            payload.hit = (a < 0.5) ? 0.0 : 1.0;
            return;
        }
        payload.hit = 1.0;
        return;
    }

    // World-space face normal from the hit triangle.
    vec3 p0 = gl_HitTriangleVertexPositionsEXT[0];
    vec3 p1 = gl_HitTriangleVertexPositionsEXT[1];
    vec3 p2 = gl_HitTriangleVertexPositionsEXT[2];
    vec3 N = normalize(cross(p1 - p0, p2 - p0));
    vec3 V = -gl_WorldRayDirectionEXT;
    if (dot(N, V) < 0.0) N = -N;

    // World-space hit position from the ray origin/direction and parametric T.
    vec3 pos = gl_WorldRayOriginEXT + gl_WorldRayDirectionEXT * gl_HitTEXT;

    // Direct (Lambert) lighting. Albedo is varied by geometry kind to visually
    // separate solid / water / vegetation; a real build would fetch the material
    // from the Materials SSBO (binding 5) using the instance/material index.
    vec3 albedo = vec3(0.7);
    if (kind == 1u)      albedo = vec3(0.20, 0.42, 0.72); // water
    else if (kind == 2u) albedo = vec3(0.25, 0.60, 0.22); // vegetation
    float ndl = max(dot(N, normalize(-ubo.lightDir.xyz)), 0.0);
    vec3 direct = albedo * (0.15 + 0.85 * ndl) * ubo.lightColor.rgb;

    payload.hit = 1.0;
    payload.normal = N;
    payload.pos = pos;
    payload.color = direct;

    if (pc.uMode == 1u) {
        // Reflection: sample the environment (or further geometry) in the
        // reflected direction and blend by a Schlick fresnel term.
        vec3 reflDir = reflect(gl_WorldRayDirectionEXT, N);
        innerPayload.hit = 0.0; innerPayload.color = vec3(0.0);
        innerPayload.normal = vec3(0.0); innerPayload.pos = vec3(0.0);
        traceRayEXT(tlas, gl_RayFlagsNoneEXT, 0xFFu, 0u, 0u, 0u, pos + N * 0.01, 0.01, reflDir, 10000.0, 1);
        float fresnel = 0.04 + 0.96 * pow(1.0 - max(dot(N, V), 0.0), 5.0);
        payload.color = mix(direct, innerPayload.color, fresnel);
    } else if (pc.uMode == 2u) {
        // Refraction / transmission: trace a refracted ray and tint by a
        // transmittance color to approximate glass/water look.
        vec3 transDir = refract(gl_WorldRayDirectionEXT, N, 0.85);
        if (dot(transDir, transDir) < 1e-4) transDir = gl_WorldRayDirectionEXT; // TIR fallback
        innerPayload.hit = 0.0; innerPayload.color = vec3(0.0);
        innerPayload.normal = vec3(0.0); innerPayload.pos = vec3(0.0);
        traceRayEXT(tlas, gl_RayFlagsNoneEXT, 0xFFu, 0u, 0u, 0u, pos - N * 0.01, 0.01, transDir, 10000.0, 1);
        vec3 tint = vec3(0.30, 0.55, 0.80);
        payload.color = mix(direct, innerPayload.color * tint, 0.6);
    }
}

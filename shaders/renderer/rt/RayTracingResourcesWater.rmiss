// Hybrid RT water pipeline: miss shader.
// Secondary-ray miss = the existing sky/environment directly (§7/§9: "On
// miss: evaluate the sky/environment directly"). Samples the sky equirect
// produced by SkyRenderer (no 360 cubemap — removed).

#version 460
#extension GL_EXT_ray_tracing : require

#include "../../types/RTPayload.glsl"
#include "../../ubo/RayTracingParams.glsl"

#include "../../includes/rt/RtParams.glsl"

layout(std140, set = 0, binding = 3) uniform RTBlock { RayTracingParams rt; };
layout(set = 0, binding = 6) uniform sampler2D skyEquirectTex;

layout(location = 0) rayPayloadInEXT RTPayload rtPayload;

void main() {
    vec3 dir = normalize(gl_WorldRayDirectionEXT);
    vec3 sky = texture(skyEquirectTex, rtDirToEquirectUV(dir)).rgb;
    rtPayload.color = sky; rtPayload.hitDistance = -1.0; // -1 marks miss; rgen maps it per-ray
    rtPayload.coarseF = 0.0; // nothing to feather on miss
}

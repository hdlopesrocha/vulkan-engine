// Radiance miss shader: evaluate the existing sky/atmospheric environment for
// primary, reflection and refraction rays that leave the scene. Never black.
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#include "rt_common.glsl"

layout(location = 0) rayPayloadInEXT RadiancePayload payload;

void main() {
    vec3 dir = gl_WorldRayDirectionEXT;
    payload.radiance = rtSkyColor(dir);
    payload.hitT = -1.0;
    payload.normal = -dir;
    payload.fresnel = 0.0;
    payload.materialId = -1;
    payload.shadow = 1.0;
}

#version 460
#extension GL_EXT_ray_tracing : enable
#include "rt_common.glsl"

// Miss for reflection/refraction rays: escape to the environment. The rgen
// reads payload.color as the reflected/transmitted environment contribution.
layout(location = 0) rayPayloadInEXT Payload payload;
void main() {
    payload.hit = 0.0;
    payload.color = environmentColor(gl_WorldRayDirectionEXT);
}

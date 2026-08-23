#version 460
#extension GL_EXT_ray_tracing : enable
#include "rt_common.glsl"

// Miss for shadow rays: no geometry was hit, so the receiver point is lit.
layout(location = 0) rayPayloadInEXT Payload payload;
void main() {
    payload.hit = 0.0;
}

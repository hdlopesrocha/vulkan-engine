// Shadow miss shader: the shadow ray escaped, the surface is lit.
#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_GOOGLE_include_directive : require
#include "rt_common.glsl"

layout(location = 1) rayPayloadInEXT ShadowPayload payload;

void main() {
    payload.occlusion = 0.0;
}

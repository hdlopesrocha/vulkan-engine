#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in uint inCornerIndex;

layout(location = 0) out float fragSdf;

#include "includes/ubo.glsl"

struct InstanceData {
    mat4 model;
    vec4 sdf0;
    vec4 sdf1;
};

layout(set = 1, binding = 0, std430) readonly buffer InstanceBuffer {
    InstanceData instances[];
};

float getCornerSdf(InstanceData inst, uint cornerIndex) {
    if (cornerIndex == 0u) return inst.sdf0.x;
    if (cornerIndex == 1u) return inst.sdf0.y;
    if (cornerIndex == 2u) return inst.sdf0.z;
    if (cornerIndex == 3u) return inst.sdf0.w;
    if (cornerIndex == 4u) return inst.sdf1.x;
    if (cornerIndex == 5u) return inst.sdf1.y;
    if (cornerIndex == 6u) return inst.sdf1.z;
    return inst.sdf1.w;
}

void main() {
    InstanceData inst = instances[gl_InstanceIndex];
    vec4 worldPos = inst.model * vec4(inPosition, 1.0);
    gl_Position = ubo.viewProjection * worldPos;
    fragSdf = getCornerSdf(inst, inCornerIndex);
}

#version 450

#include "includes/locations.glsl"

layout(location = ATTR_POS) in vec3 inPosition;
layout(location = ATTR_COLOR) in uint inCornerIndex;

layout(location = VARY_SDF) out float fragSdf;
layout(location = VARY_BRUSHPATCH) flat out int fragBrushIndex;

#include "includes/ubo.glsl"

struct InstanceData {
    mat4 model;
    vec4 sdf0;
    vec4 sdf1;
    vec4 meta; // meta.x = brushIndex (stored as float)
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
    fragBrushIndex = int(inst.meta.x + 0.5);
}

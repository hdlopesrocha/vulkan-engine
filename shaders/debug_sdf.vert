#version 450

#include "types/DebugSdfInstanceData.glsl"
#include "types/DebugSdfInstanceDataNamed.glsl"

#include "includes/locations.glsl"

layout(location = ATTR_POS) in vec3 inPosition;
layout(location = ATTR_COLOR) in uint inCornerIndex;

layout(location = VARY_SDF) out float fragSdf;
layout(location = VARY_BRUSHPATCH) flat out int fragBrushIndex;

#include "includes/ubo.glsl"


layout(set = 1, binding = 0, std430) readonly buffer InstanceBuffer {
    DebugSdfInstanceData instances[];
};


DebugSdfInstanceDataNamed instanceNamed(DebugSdfInstanceData p) {
    DebugSdfInstanceDataNamed n;
    n.model = p.model;
    n.sdfCorner0 = p.sdf0.x;
    n.sdfCorner1 = p.sdf0.y;
    n.sdfCorner2 = p.sdf0.z;
    n.sdfCorner3 = p.sdf0.w;
    n.sdfCorner4 = p.sdf1.x;
    n.sdfCorner5 = p.sdf1.y;
    n.sdfCorner6 = p.sdf1.z;
    n.sdfCorner7 = p.sdf1.w;
    n.brushIndex = int(p.meta.x + 0.5);
    return n;
}

float getCornerSdf(DebugSdfInstanceDataNamed inst, uint cornerIndex) {
    if (cornerIndex == 0u) return inst.sdfCorner0;
    if (cornerIndex == 1u) return inst.sdfCorner1;
    if (cornerIndex == 2u) return inst.sdfCorner2;
    if (cornerIndex == 3u) return inst.sdfCorner3;
    if (cornerIndex == 4u) return inst.sdfCorner4;
    if (cornerIndex == 5u) return inst.sdfCorner5;
    if (cornerIndex == 6u) return inst.sdfCorner6;
    return inst.sdfCorner7;
}

void main() {
    DebugSdfInstanceDataNamed inst = instanceNamed(instances[gl_InstanceIndex]);
    vec4 worldPos = inst.model * vec4(inPosition, 1.0);
    gl_Position = ubo.viewProjection * worldPos;
    fragSdf = getCornerSdf(inst, inCornerIndex);
    fragBrushIndex = inst.brushIndex;
}

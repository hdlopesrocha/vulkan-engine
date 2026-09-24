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

// Named view over the packed InstanceData - same data, descriptive names. The builder below is the
// only place the packed component letters are read; every other access uses the
// named attributes.
// sdf0/sdf1 pack the 8 box-corner SDF values, meta.x carries the brush index.
struct InstanceDataNamed {
    mat4 model;
    float sdfCorner0;
    float sdfCorner1;
    float sdfCorner2;
    float sdfCorner3;
    float sdfCorner4;
    float sdfCorner5;
    float sdfCorner6;
    float sdfCorner7;
    int brushIndex;
};

InstanceDataNamed instanceNamed(InstanceData p) {
    InstanceDataNamed n;
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

float getCornerSdf(InstanceDataNamed inst, uint cornerIndex) {
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
    InstanceDataNamed inst = instanceNamed(instances[gl_InstanceIndex]);
    vec4 worldPos = inst.model * vec4(inPosition, 1.0);
    gl_Position = ubo.viewProjection * worldPos;
    fragSdf = getCornerSdf(inst, inCornerIndex);
    fragBrushIndex = inst.brushIndex;
}

#version 450

#include "includes/locations.glsl"

layout(location = ATTR_POS) in vec3 inPosition;
layout(location = ATTR_NORMAL) in vec3 inNormal;
layout(location = ATTR_UV) in vec3 inTexCoord;

layout(location = VARY_UV) out vec3 fragTexCoord;
layout(location = VARY_COLOR) out vec3 fragColor;

#include "includes/ubo.glsl"

struct InstanceData {
    mat4 model;
    vec4 color;  // vec4 for proper alignment
};

layout(set = 1, binding = 2, std430) readonly buffer InstanceBuffer {
    InstanceData instances[];
};

// Named view over the packed InstanceData - same data, descriptive names. The builder below is the
// only place the packed component letters are read; every other access uses the
// named attributes.
struct InstanceDataNamed {
    mat4 model;
    vec3 color;
};

InstanceDataNamed instanceNamed(InstanceData p) {
    InstanceDataNamed n;
    n.model = p.model;
    n.color = p.color.rgb;
    return n;
}

void main() {
    InstanceDataNamed inst = instanceNamed(instances[gl_InstanceIndex]);
    vec4 worldPos = inst.model * vec4(inPosition, 1.0);
    gl_Position = ubo.viewProjection * worldPos;
    fragTexCoord = inTexCoord;
    fragColor = inst.color;
}

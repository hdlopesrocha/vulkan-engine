#version 450

#include "../../types/DebugCubeInstanceData.glsl"
#include "../../types/DebugCubeInstanceDataNamed.glsl"

#include "../../includes/Locations.glsl"

layout(location = ATTR_POS) in vec3 inPosition;
layout(location = ATTR_NORMAL) in vec3 inNormal;
layout(location = ATTR_UV) in vec3 inTexCoord;

layout(location = VARY_UV) out vec3 fragTexCoord;
layout(location = VARY_COLOR) out vec3 fragColor;

#include "../../includes/SceneBindings.glsl"


layout(set = 1, binding = 2, std430) readonly buffer InstanceBuffer {
    DebugCubeInstanceData instances[];
};


DebugCubeInstanceDataNamed instanceNamed(DebugCubeInstanceData p) {
    DebugCubeInstanceDataNamed n;
    n.model = p.model;
    n.color = p.color.rgb;
    return n;
}

void main() {
    DebugCubeInstanceDataNamed inst = instanceNamed(instances[gl_InstanceIndex]);
    vec4 worldPos = inst.model * vec4(inPosition, 1.0);
    gl_Position = ubo.viewProjection * worldPos;
    fragTexCoord = inTexCoord;
    fragColor = inst.color;
}

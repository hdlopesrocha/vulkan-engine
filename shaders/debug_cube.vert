#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec3 inTexCoord;

layout(location = 0) out vec3 fragTexCoord;
layout(location = 1) out vec3 fragColor;

layout(set = 0, binding = 0) uniform UBO {
    mat4 viewProjection;
    vec4 viewPos;
    vec4 lightDir;
    vec4 lightColor;
    vec4 materialFlags;
    mat4 lightSpaceMatrix;
    vec4 shadowEffects;
    vec4 debugParams;
    vec4 triplanarSettings;
    vec4 tessParams;
    vec4 passParams;
} ubo;

struct InstanceData {
    mat4 model;
    vec4 color;  // vec4 for proper alignment
};

layout(set = 1, binding = 2, std430) readonly buffer InstanceBuffer {
    InstanceData instances[];
};

void main() {
    InstanceData inst = instances[gl_InstanceIndex];
    vec4 worldPos = inst.model * vec4(inPosition, 1.0);
    gl_Position = ubo.viewProjection * worldPos;
    fragTexCoord = inTexCoord;
    fragColor = inst.color.rgb;
}

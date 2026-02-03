#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in int inTexIndex;

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) out flat int fragTexIndex;


layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 viewProj;
} ubo;

layout(push_constant) uniform PushConstants {
    float billboardScale;
};

layout(location = 4) in vec3 instancePosition;

void main() {
    // Billboard: orient quad to face camera in geometry shader, so here just pass position
    vec3 pos = inPosition * billboardScale + instancePosition;
    gl_Position = ubo.viewProj * ubo.model * vec4(pos, 1.0);
    fragTexCoord = inTexCoord;
    fragTexIndex = inTexIndex;
}

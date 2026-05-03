#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in int inTexIndex;
layout(location = 4) in vec4 instanceData; // .xyz = world pos, .w = billboard index

layout(location = 0) out vec2 fragTexCoord;
layout(location = 1) flat out int fragTexIndex;
layout(location = 2) out vec3 fragWorldPos;

// Must match SolidParamsUBO — only read the first two fields.
layout(set = 0, binding = 0) uniform SolidParamsUBO {
    mat4 viewProjection;
    vec4 viewPos;
} ubo;

layout(push_constant) uniform PushConstants {
    float billboardScale;
};

void main() {
    // inPosition is always (0,0,0) for the base vertex; world position = instanceData.xyz.
    vec3 worldPos = instanceData.xyz;
    gl_Position = ubo.viewProjection * vec4(worldPos, 1.0);
    fragTexCoord = inTexCoord;
    fragTexIndex = int(round(instanceData.w)); // per-instance billboard index
    fragWorldPos = worldPos;
}

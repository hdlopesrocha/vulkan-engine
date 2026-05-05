#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in int  inTexIndex;
layout(location = 4) in vec4 instanceData; // xyz=world pos, w=billboard index

layout(location = 0) out vec3 outWorldPos;
layout(location = 1) flat out int outBillboardIndex;

// Must match SolidParamsUBO — only read the first two fields.
layout(set = 0, binding = 0) uniform SolidParamsUBO {
    mat4 viewProjection;
    vec4 viewPos;
} ubo;

layout(push_constant) uniform PushConstants {
    float billboardScale;
    float windEnabled;
    float windTime;
    float impostorDistance;
    vec4 windDirAndStrength;
    vec4 windNoise;
    vec4 windShape;
    vec4 windTurbulence;
    vec4 densityParams;
    vec4 cameraPosAndFalloff;
};

void main() {
    vec3 worldPos = instanceData.xyz;
    // Pass world position as clip coords placeholder; geo shader does the real projection.
    gl_Position = ubo.viewProjection * vec4(worldPos, 1.0);
    outWorldPos       = worldPos;
    outBillboardIndex = int(round(instanceData.w));
}

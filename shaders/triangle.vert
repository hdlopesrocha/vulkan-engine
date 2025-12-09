#version 450

// UBO layout must match the CPU-side UniformObject (std140-like):
// mat4 mvp; vec4 lightDir; vec4 lightColor;
layout(binding = 0) uniform UBO {
    mat4 mvp;
    vec4 lightDir;
    vec4 lightColor;
} ubo;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragUV;
layout(location = 2) out vec3 fragNormal;

void main() {
    fragColor = inColor;
    fragUV = inUV;
    // approximate normal from position (cube is centered at origin)
    fragNormal = normalize(inPos);
    // apply MVP transform from UBO
    gl_Position = ubo.mvp * vec4(inPos, 1.0);
}

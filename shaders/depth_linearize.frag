#version 450

layout(binding = 0) uniform sampler2D depthTex;

layout(push_constant) uniform PC {
    float zNear;
    float zFar;
    float mode; // 0.0 = perspective linearize, 1.0 = passthrough
} pc;

layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outColor;

// Linearize perspective depth from non-linear depth buffer value
float LinearizeDepth(float d, float near, float far) {
    // d is the normalized depth value in [0,1] as written to the depth buffer
    // Convert to linear eye-space depth using the common formula
    return (near * far) / (far - d * (far - near));
}

void main() {
    float d = texture(depthTex, uv).r;
    float v = 0.0;
    if (pc.mode >= 0.5) {
        // Passthrough: treat sampled depth as already linear (e.g. shadow maps)
        v = clamp(d, 0.0, 1.0);
    } else {
        float linear = LinearizeDepth(d, pc.zNear, pc.zFar);
        v = clamp(linear / pc.zFar, 0.0, 1.0);
    }
    outColor = vec4(v, v, v, 1.0);
}

#version 450

// Separable Gaussian blur for EVSM shadow maps.
// Push constant .x: 0 = horizontal, 1 = vertical

layout(set = 0, binding = 0) uniform sampler2D evsmTexture;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform PC {
    float direction; // 0 = horizontal, 1 = vertical
} pc;

// 7-tap Gaussian kernel (sigma = 1.5, normalized)
// Reduced blur to preserve alpha-rejected vegetation shadow detail.
const float KERNEL[7] = float[](
    0.0300, 0.1047, 0.2225, 0.2856,
    0.2225, 0.1047, 0.0300
);
const int RADIUS = 3;

void main() {
    ivec2 texSize = textureSize(evsmTexture, 0);
    vec2 uv = gl_FragCoord.xy / vec2(texSize);

    vec2 step;
    if (pc.direction < 0.5) {
        step = vec2(1.0 / texSize.x, 0.0);
    } else {
        step = vec2(0.0, 1.0 / texSize.y);
    }

    vec4 result = vec4(0.0);
    for (int i = -RADIUS; i <= RADIUS; ++i) {
        vec2 offset = float(i) * step;
        result += KERNEL[i + RADIUS] * texture(evsmTexture, uv + offset);
    }

    outColor = result;
}

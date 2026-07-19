#version 450

// Separable Gaussian blur for EVSM shadow maps.
// Push constant .x: 0 = horizontal, 1 = vertical

layout(set = 0, binding = 0) uniform sampler2D evsmTexture;

layout(location = 0) out vec2 outColor;

layout(push_constant) uniform PC {
    float direction; // 0 = horizontal, 1 = vertical
} pc;

// 3-tap Gaussian kernel (sigma = 0.5, normalized)
// Minimal blur to avoid light-bleeding halo on water.
const float KERNEL[3] = float[](
    0.2747, 0.4506, 0.2747
);
const int RADIUS = 1;

void main() {
    ivec2 texSize = textureSize(evsmTexture, 0);
    vec2 uv = gl_FragCoord.xy / vec2(texSize);

    vec2 step;
    if (pc.direction < 0.5) {
        step = vec2(1.0 / texSize.x, 0.0);
    } else {
        step = vec2(0.0, 1.0 / texSize.y);
    }

    vec2 result = vec2(0.0);
    for (int i = -RADIUS; i <= RADIUS; ++i) {
        vec2 offset = float(i) * step;
        result += KERNEL[i + RADIUS] * texture(evsmTexture, uv + offset).xy;
    }

    outColor = result;
}

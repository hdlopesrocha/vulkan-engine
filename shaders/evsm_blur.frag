#version 450

// Separable Gaussian blur for EVSM shadow maps.
// Push constant .x: 0 = horizontal, 1 = vertical

layout(set = 0, binding = 0) uniform sampler2D evsmTexture;

layout(location = 0) out vec2 outColor;

layout(push_constant) uniform PC {
    float direction; // 0 = horizontal, 1 = vertical
} pc;

// 9-tap separable Gaussian (sigma = 2.0, normalized). The previous 3-tap
// kernel left EVSM moment edges essentially unblurred, so shadows rendered
// blocky/pixelated. A wider kernel smooths the Chebyshev transition and the
// cascade silhouettes without washing the shadow out.
const float KERNEL[9] = float[](
    0.0218, 0.0671, 0.1254, 0.1824, 0.2066,
    0.1824, 0.1254, 0.0671, 0.0218
);
const int RADIUS = 4;

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

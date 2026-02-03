#version 450

layout(set = 0, binding = 0) uniform sampler2D sceneColorTex;
layout(set = 0, binding = 1) uniform sampler2D sceneDepthTex; // unused, reserved
layout(set = 0, binding = 2) uniform sampler2D waterDepthTex; // unused, reserved
layout(set = 0, binding = 3) uniform sampler2D waterNormalTex; // unused, reserved
layout(set = 0, binding = 4) uniform sampler2D waterMaskTex; // unused, reserved

layout(set = 0, binding = 5) uniform WaterUBO {
    mat4 viewProjection;
    mat4 invViewProjection;
    vec4 viewPos;
    vec4 waterParams1;
    vec4 waterParams2;
    vec4 shallowColor;
    vec4 deepColor;
    vec4 screenSize;     // xy = width,height
    vec4 noisePersistence;
} ubo;

layout(location = 0) out vec4 outColor;

void main() {
    // Convert fragment position to UV
    vec2 uv = gl_FragCoord.xy / ubo.screenSize.xy;
    outColor = texture(sceneColorTex, uv);
}

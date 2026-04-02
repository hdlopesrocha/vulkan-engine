#version 450

layout(set = 0, binding = 0) uniform sampler2D sceneColorTex;
layout(set = 0, binding = 1) uniform sampler2D sceneDepthTex;
layout(set = 0, binding = 2) uniform sampler2D waterColorTex;  // Water geometry pass output (attachment 0)
layout(set = 0, binding = 3) uniform sampler2D waterNormalTex;
layout(set = 0, binding = 4) uniform sampler2D waterMaskTex;

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

    vec4 sceneColor = texture(sceneColorTex, uv);
    vec4 waterColor = texture(waterColorTex, uv);

    // The water geometry pass clears attachment 0 to (1000.0, 0.0, 0.0, 0.0).
    // Water fragments write valid colors with alpha = 1.0.
    // Use alpha to detect water pixels and composite over the scene.
    float waterAlpha = waterColor.a;

    outColor = mix(sceneColor, vec4(waterColor.rgb, 1.0), waterAlpha);
}

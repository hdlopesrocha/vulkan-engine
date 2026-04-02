#version 450

// Final compositing pass: Sky (equirectangular) + Solid + Water
// Background pixels (depth == 1.0) are filled with sky sampled from equirect map.
// Solid geometry pixels are used as-is.
// Water is composited on top using water alpha.

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

layout(set = 0, binding = 6) uniform sampler2D sceneSkyTex;

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265358979;

// Convert a 3D direction to equirectangular UV [0,1]
vec2 dirToEquirectUV(vec3 dir) {
    vec2 uv;
    uv.x = atan(dir.z, dir.x) / (2.0 * PI) + 0.5;
    uv.y = acos(clamp(dir.y, -1.0, 1.0)) / PI;
    return uv;
}

void main() {
    // Convert fragment position to UV
    vec2 uv = gl_FragCoord.xy / ubo.screenSize.xy;

    // Sample scene (solid) color and depth
    vec4 sceneColor = texture(sceneColorTex, uv);
    float depth = texture(sceneDepthTex, uv).r;

    // Reconstruct world-space view direction for background sky sampling
    // Use NDC coordinates with depth to get clip-space position, then inverse transform
    vec2 ndc = uv * 2.0 - 1.0;
    vec4 clipPos = vec4(ndc, depth, 1.0);
    vec4 worldPos = ubo.invViewProjection * clipPos;
    worldPos /= worldPos.w;
    vec3 viewDir = normalize(worldPos.xyz - ubo.viewPos.xyz);

    // Sample sky from equirectangular map using the view direction
    vec2 skyUV = dirToEquirectUV(viewDir);
    vec3 skyColor = texture(sceneSkyTex, skyUV).rgb;

    // Composite layers:
    // 1. Start with sky as the base background
    // 2. Where solid geometry exists (depth < 1.0), use scene color
    // 3. Composite water on top using water alpha
    vec3 baseColor = skyColor;

    // Blend in solid geometry where depth < 1.0
    // Use a tight threshold to avoid floating-point edge issues
    float isSolid = (depth < 0.9999) ? 1.0 : 0.0;
    baseColor = mix(baseColor, sceneColor.rgb, isSolid);

    // Composite water on top
    vec4 waterColor = texture(waterColorTex, uv);
    float waterAlpha = waterColor.a;
    vec3 finalColor = mix(baseColor, waterColor.rgb, waterAlpha);

    outColor = vec4(finalColor, 1.0);
}

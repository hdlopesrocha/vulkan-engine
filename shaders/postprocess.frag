#version 450

// Final compositing pass: Sky (equirectangular) + Solid + Water
// Background pixels (depth == 1.0) are filled with sky sampled from equirect map.
// Solid geometry pixels are used as-is.
// Water is composited on top using water alpha.

layout(set = 0, binding = 0) uniform sampler2D sceneColorTex;
layout(set = 0, binding = 1) uniform sampler2D sceneDepthTex;
layout(set = 0, binding = 2) uniform sampler2D waterColorTex;  // Water geometry pass output (attachment 0)

layout(set = 0, binding = 5) uniform WaterUBO {
    mat4 viewProjection;
    mat4 invViewProjection;
    vec4 viewPos;
    vec4 screenSize;     // xy = width,height
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

    // Blend in solid geometry where scene alpha > 0 (solid writes alpha=1, clear is alpha=0).
    // Previous depth-based threshold (depth < 0.9999) incorrectly treated far
    // solid fragments as sky because the non-linear depth buffer clusters
    // values near 1.0 at large distances (e.g. at d=1000 with n=0.1/f=8092,
    // depth ≈ 0.9999 already).
    float isSolid = sceneColor.a;
    baseColor = mix(baseColor, sceneColor.rgb, isSolid);

    // Composite water on top
    vec4 waterColor = texture(waterColorTex, uv);
    float waterAlpha = waterColor.a;
    vec3 finalColor = mix(baseColor, waterColor.rgb, waterAlpha);

    outColor = vec4(finalColor, 1.0);
}

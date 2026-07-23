#version 450

#include "includes/locations.glsl"

// Final compositing pass: Sky + Solid + Water + Brush
// Background pixels are filled with sky.
// Solid geometry pixels are used as-is.
// Water is composited on top using water alpha.
// Brush is composited on top with depth testing against both scene and water depth.

layout(set = 0, binding = 0) uniform sampler2D sceneColorTex;
layout(set = 0, binding = 1) uniform sampler2D sceneDepthTex;
layout(set = 0, binding = 2) uniform sampler2D waterColorTex;
layout(set = 0, binding = 3) uniform sampler2D brushColorTex;
layout(set = 0, binding = 4) uniform sampler2D brushDepthTex;

layout(set = 0, binding = 5) uniform WaterUBO {
    mat4 viewProjection;
    mat4 invViewProjection;
    vec4 viewPos;
    vec4 screenSize;
    float brushAlpha;
} ubo;

layout(set = 0, binding = 6) uniform sampler2D sceneSkyTex;
layout(set = 0, binding = 7) uniform sampler2D waterGeomDepthTex;

layout(location = FRAG_OUT_COLOR) out vec4 outColor;

const float PI = 3.14159265358979;

vec2 dirToEquirectUV(vec3 dir) {
    vec2 uv;
    uv.x = atan(dir.z, dir.x) / (2.0 * PI) + 0.5;
    uv.y = acos(clamp(dir.y, -1.0, 1.0)) / PI;
    return uv;
}

void main() {
    vec2 uv = gl_FragCoord.xy / ubo.screenSize.xy;

    vec4 sceneColor = texture(sceneColorTex, uv);
    float sceneDepth = texture(sceneDepthTex, uv).r;

    vec2 ndc = uv * 2.0 - 1.0;
    vec4 clipPos = vec4(ndc, sceneDepth, 1.0);
    vec4 worldPos = ubo.invViewProjection * clipPos;
    worldPos /= worldPos.w;
    vec3 viewDir = normalize(worldPos.xyz - ubo.viewPos.xyz);

    vec2 skyUV = dirToEquirectUV(viewDir);
    vec3 skyColor = texture(sceneSkyTex, skyUV).rgb;

    // 1. Sky background
    vec3 baseColor = skyColor;
    // 2. Solid geometry (alpha > 0)
    float isSolid = sceneColor.a;
    baseColor = mix(baseColor, sceneColor.rgb, isSolid);
    // 3. Water on top
    vec4 waterColor = texture(waterColorTex, uv);
    float waterAlpha = waterColor.a;
    vec3 afterWater = mix(baseColor, waterColor.rgb, waterAlpha);

    vec3 finalColor = afterWater;

    // 4. Brush overlay with depth test against scene + water
    vec4 brushColor = texture(brushColorTex, uv);
    if (brushColor.a > 0.0) {
        float brushDepth = texture(brushDepthTex, uv).r;

        // Scene solid depth is the reference obstacle depth
        float obstacleDepth = sceneDepth;

        // Where water geometry exists, use water surface depth as obstacle
        // (water is always in front of the solid geometry behind it)
        float waterGeomDepth = texture(waterGeomDepthTex, uv).r;
        if (waterGeomDepth < 1.0) {
            // Water geometry is present — use the closer of the two
            obstacleDepth = min(obstacleDepth, waterGeomDepth);
        }

        // Brush is visible only where it is in front of the closest obstacle
        if (brushDepth < obstacleDepth) {
            // Blend brush over the current composite using the selected opacity
            finalColor = mix(finalColor, brushColor.rgb, ubo.brushAlpha);
        }
    }

    outColor = vec4(finalColor, 1.0);
}

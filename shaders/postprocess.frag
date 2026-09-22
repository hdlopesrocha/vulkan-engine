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
    float brushMode;         // 0=overlay, 2=PAINT (replace solid texture)
} ubo;

layout(set = 0, binding = 6) uniform sampler2D sceneSkyTex;
layout(set = 0, binding = 7) uniform sampler2D waterGeomDepthTex;
layout(set = 0, binding = 8) uniform sampler2D brushBackFaceDepthTex;
// Vegetation offscreen color + depth (decoupled from the solid pass so it can be
// rendered on a parallel async command buffer). Occlusion against solid geometry
// is resolved here by testing vegetation depth against the solid scene depth.
layout(set = 0, binding = 9) uniform sampler2D vegColorTex;
layout(set = 0, binding = 10) uniform sampler2D vegDepthTex;
// SDF debug cubes offscreen color + depth (own async command buffer / queue).
layout(set = 0, binding = 11) uniform sampler2D sdfColorTex;
layout(set = 0, binding = 12) uniform sampler2D sdfDepthTex;
// Mesh bounding boxes offscreen color + depth (own async command buffer / queue).
layout(set = 0, binding = 13) uniform sampler2D bboxColorTex;
layout(set = 0, binding = 14) uniform sampler2D bboxDepthTex;
// Water refraction/tint blur inputs (color attachments 1 and 2 of the water
// geometry pass):
//  body   = RGB refraction + tint body (pre-reflection), A = body weight
//           (coverage times the body's share of the final mix). Only this
//           body is blurred, so the reflection stays sharp.
//  column = R measured water depth (m), G = per-material blur radius (pixels,
//           0 = crisp), B/A unused.
layout(set = 0, binding = 15) uniform sampler2D waterBodyTex;
layout(set = 0, binding = 16) uniform sampler2D waterColumnTex;

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

    // 2.5 Vegetation offscreen (decoupled from the solid pass). Composite it over
    // the solid, but hide fragments the solid geometry occludes (a solid surface
    // in front of the vegetation). The vegetation depth is also tracked as an
    // obstacle for the water/brush occlusion tests below.
    float vegDepth = texture(vegDepthTex, uv).r;
    bool vegPresent = (vegDepth < 1.0);
    float obstacleDepth = sceneDepth;
    if (vegPresent) {
        obstacleDepth = min(obstacleDepth, vegDepth);
        vec4 vegColor = texture(vegColorTex, uv);
        if (vegColor.a > 0.0 && !(sceneDepth < vegDepth)) {
            baseColor = mix(baseColor, vegColor.rgb, vegColor.a);
        }
    }
    // 3. Water on top
    vec4 waterColor = texture(waterColorTex, uv);
    float waterAlpha = waterColor.a;
    // Depth-guided water blur, performed here in the final pass. Only the
    // REFRACTION + TINT body (waterBodyTex.rgb) is blurred and re-inserted
    // with its stored weight (waterBodyTex.a = coverage * body share of the
    // mix), so the reflection lobe, specular highlights, caustics and foam
    // stay sharp. The per-material blur radius (waterColumnTex.g, computed by
    // the water shader from the measured depth and the layer's blur params)
    // sets the disc size, and a bilateral depth weight keeps the blur from
    // smearing across depth edges.
    if (waterAlpha > 0.0) {
        vec4 bodyCenter = textureLod(waterBodyTex, uv, 0.0);
        vec2 columnCenter = textureLod(waterColumnTex, uv, 0.0).rg;
        float blurPx = columnCenter.g;
        if (blurPx > 0.5 && bodyCenter.a > 1e-4) {
            vec2 texel = ubo.screenSize.zw;
            // 9-tap disc: center + 4 axis at r + 4 diagonal at 0.7r.
            const vec2 kOffs[8] = vec2[8](
                vec2( 1.0,  0.0), vec2(-1.0,  0.0), vec2( 0.0,  1.0), vec2( 0.0, -1.0),
                vec2( 0.7,  0.7), vec2(-0.7,  0.7), vec2( 0.7, -0.7), vec2(-0.7, -0.7));
            vec3 accBody = bodyCenter.rgb * bodyCenter.a;
            float wsum = bodyCenter.a;
            for (int i = 0; i < 8; ++i) {
                vec2 suv = clamp(uv + kOffs[i] * texel * blurPx, vec2(0.001), vec2(0.999));
                vec4 s = textureLod(waterBodyTex, suv, 0.0);
                float columnS = textureLod(waterColumnTex, suv, 0.0).r;
                // Bilateral depth weight: preserve depth edges (shoreline).
                float depthW = exp(-abs(columnS - columnCenter.r) * 0.5);
                float w = s.a * depthW;
                accBody += s.rgb * w;
                wsum += w;
            }
            vec3 blurredBody = accBody / max(wsum, 1e-4);
            // Replace only the body's share of the composed pixel: the
            // reflection and surface terms in waterColor are untouched.
            waterColor.rgb = max(waterColor.rgb + bodyCenter.a * (blurredBody - bodyCenter.rgb),
                                 vec3(0.0));
        }
    }
    // Occlusion against solids (and vegetation): the water pass no longer samples
    // the solid depth texture (so it can be recorded/rendered independently of the
    // solid pass), so the depth test against solid geometry is resolved here
    // instead. If a solid surface (or vegetation) is in front of the water
    // surface, hide the water fragment.
    {
        float waterGeomDepth = texture(waterGeomDepthTex, uv).r;
        if (waterGeomDepth < 1.0 && obstacleDepth < waterGeomDepth) {
            waterAlpha = 0.0;
        }
    }
    vec3 afterWater = mix(baseColor, waterColor.rgb, waterAlpha);

    vec3 finalColor = afterWater;

    // 4. Brush compositing
    vec4 brushColor = texture(brushColorTex, uv);
    if (brushColor.a > 0.0) {
        float brushDepth = texture(brushDepthTex, uv).r;

        // Brush geometry overlay: depth-test brush against scene + water + veg, same for all modes
        {
            if (brushDepth < obstacleDepth) {
                finalColor = mix(finalColor, brushColor.rgb, ubo.brushAlpha);
            }
        }
    }

    // 5. Debug overlays (SDF cubes + mesh bounding boxes) — composited after the
    // brush so they sit on top, but still occluded by solid + vegetation geometry.
    // Both render to their own offscreen depth; we hide a debug fragment that is
    // behind the current obstacle (solid or vegetation) surface.
    float sdfDepth = texture(sdfDepthTex, uv).r;
    if (sdfDepth < 1.0 && !(obstacleDepth < sdfDepth)) {
        vec4 sdfColor = texture(sdfColorTex, uv);
        if (sdfColor.a > 0.0) {
            finalColor = mix(finalColor, sdfColor.rgb, sdfColor.a);
        }
    }
    float bboxDepth = texture(bboxDepthTex, uv).r;
    if (bboxDepth < 1.0 && !(obstacleDepth < bboxDepth)) {
        vec4 bboxColor = texture(bboxColorTex, uv);
        if (bboxColor.a > 0.0) {
            finalColor = mix(finalColor, bboxColor.rgb, bboxColor.a);
        }
    }

    outColor = vec4(finalColor, 1.0);
}

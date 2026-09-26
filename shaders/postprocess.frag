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
    float waterBlurEnabled;  // 1 = body/column written this frame, fetch/blur allowed
} uboPacked;

// Named view over the packed WaterUBO - same data, descriptive names. The builder below is the
// only place the packed component letters are read; every other access uses the
// named attributes.
struct WaterFrameNamed {
    mat4 viewProjection;
    mat4 invViewProjection;
    vec3 viewPosition;
    vec2 screenSize;
    vec2 invScreenSize;
    float brushAlpha;
    float brushMode;
    float waterBlurEnabled;
};

WaterFrameNamed waterFrameNamed() {
    WaterFrameNamed n;
    n.viewProjection = uboPacked.viewProjection;
    n.invViewProjection = uboPacked.invViewProjection;
    n.viewPosition = uboPacked.viewPos.xyz;
    n.screenSize = uboPacked.screenSize.xy;
    n.invScreenSize = uboPacked.screenSize.zw;
    n.brushAlpha = uboPacked.brushAlpha;
    n.brushMode = uboPacked.brushMode;
    n.waterBlurEnabled = uboPacked.waterBlurEnabled;
    return n;
}

WaterFrameNamed ubo = waterFrameNamed();

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
    vec2 uv = gl_FragCoord.xy / ubo.screenSize;

    vec4 sceneColor = texture(sceneColorTex, uv);
    float sceneDepth = texture(sceneDepthTex, uv).r;

    vec2 ndc = uv * 2.0 - 1.0;
    vec4 clipPos = vec4(ndc, sceneDepth, 1.0);
    vec4 worldPos = ubo.invViewProjection * clipPos;
    worldPos /= worldPos.w;
    vec3 viewDir = normalize(worldPos.xyz - ubo.viewPosition);

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
        vec4 vegColor = textureLod(vegColorTex, uv, 0.0);
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
    // H4: waterBlurEnabled is false when no layer needs this blur, i.e. the
    // water pass bound the single-attachment variant and never wrote the
    // body/column targets. The whole block is then skipped so no body/column
    // fetch is executed; the alpha/occlusion path below is unaffected.
    if (waterAlpha > 0.0 && ubo.waterBlurEnabled > 0.5) {
        vec4 bodyCenter = textureLod(waterBodyTex, uv, 0.0);
        vec2 columnCenter = textureLod(waterColumnTex, uv, 0.0).rg;
        float blurPx = columnCenter.g;
        if (blurPx > 0.5 && bodyCenter.a > 1e-4) {
            vec2 texel = ubo.invScreenSize;
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
    // The water geometry depth target may be smaller than the screen
    // (Settings::waterRenderScale), so a single bilinear tap would interpolate
    // across the water/clear edge and report the water FARTHER than it is -
    // which would hide water that is actually in front of the obstacle. Take the
    // CLOSEST of the 2x2 taps instead: the closer depth keeps the water, the
    // safe direction. Skipped where there is no water at all, so non-water
    // pixels pay nothing.
    // Only valid when the water pass actually wrote its geometry-depth target:
    // the single-attachment (body/column-less) variant that Minimal binds never
    // writes it, so this test compared against whatever the target last held
    // and hid the water entirely - black water in Minimal, correct in Maximum,
    // which is the signature that led here. waterBlurEnabled is exactly "the
    // aux attachments were written this frame" (H4).
    if (waterAlpha > 0.0 && ubo.waterBlurEnabled > 0.5) {
        vec2 wtexel = 1.0 / vec2(textureSize(waterGeomDepthTex, 0));
        // M9: explicit LOD (divergent flow — implicit derivatives are
        // undefined here and go wrong exactly at water silhouettes).
        float wd00 = textureLod(waterGeomDepthTex, clamp(uv + vec2(-wtexel.x, -wtexel.y), vec2(0.0), vec2(1.0)), 0.0).r;
        float wd10 = textureLod(waterGeomDepthTex, clamp(uv + vec2( wtexel.x, -wtexel.y), vec2(0.0), vec2(1.0)), 0.0).r;
        float wd01 = textureLod(waterGeomDepthTex, clamp(uv + vec2(-wtexel.x,  wtexel.y), vec2(0.0), vec2(1.0)), 0.0).r;
        float wd11 = textureLod(waterGeomDepthTex, clamp(uv + vec2( wtexel.x,  wtexel.y), vec2(0.0), vec2(1.0)), 0.0).r;
        float waterGeomDepth = min(min(wd00, wd10), min(wd01, wd11));
        if (waterGeomDepth < 1.0 && obstacleDepth < waterGeomDepth) {
            waterAlpha = 0.0;
        }
    }
    vec3 afterWater = mix(baseColor, waterColor.rgb, waterAlpha);

    vec3 finalColor = afterWater;

    // 4. Brush compositing
    vec4 brushColor = texture(brushColorTex, uv);
    if (brushColor.a > 0.0) {
        // M9: explicit LOD (divergent flow, same argument as above).
        float brushDepth = textureLod(brushDepthTex, uv, 0.0).r;

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
        // M9: explicit LOD (divergent flow, same argument as above).
        vec4 sdfColor = textureLod(sdfColorTex, uv, 0.0);
        if (sdfColor.a > 0.0) {
            finalColor = mix(finalColor, sdfColor.rgb, sdfColor.a);
        }
    }
    float bboxDepth = texture(bboxDepthTex, uv).r;
    if (bboxDepth < 1.0 && !(obstacleDepth < bboxDepth)) {
        // M9: explicit LOD (divergent flow, same argument as above).
        vec4 bboxColor = textureLod(bboxColorTex, uv, 0.0);
        if (bboxColor.a > 0.0) {
            finalColor = mix(finalColor, bboxColor.rgb, bboxColor.a);
        }
    }

    outColor = vec4(finalColor, 1.0);
}

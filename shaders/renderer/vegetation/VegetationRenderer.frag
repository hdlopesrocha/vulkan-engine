#version 450

#include "../../includes/Locations.glsl"

// Bladed grass color: procedural root->tip gradient + per-variant palette,
// ambient + Lambert + Blinn-Phong, hard CSM shadows + cloud shadows.
// Opaque triangles: no alpha discard (depth prepass writes every blade pixel).

layout(location = VARY_UV) in vec3 inTexCoord;  // x=side, y=heightT, z=variant
layout(location = VARY_BRUSHPATCH) in flat int inBrushIndex;
layout(location = VARY_POSWORLD) in      vec3 inWorldPos;
layout(location = VARY_PLANE_NORMAL) in flat vec3 inPlaneNormal;
layout(location = VARY_POSLIGHT) in flat vec3 inTangentWS;
layout(location = VARY_TANGENTWS) in flat vec3 inLightDirN;
layout(location = FRAG_OUT_COLOR) out vec4 outColor;

#include "../../includes/SceneBindings.glsl"
#include "../../includes/DebugModes.glsl"
#include "../../includes/sky/SkyView.glsl"
#include "../../includes/noise/Perlin.glsl"
#include "../../includes/sky/Clouds.glsl"
#include "../../includes/vegetation/BladeGrass.glsl"

layout(set = 0, binding = 4) uniform sampler2D shadowMap;
layout(set = 0, binding = 8) uniform sampler2D shadowMap1;
layout(set = 0, binding = 9) uniform sampler2D shadowMap2;

// Legacy billboard texture set (set=1) is retained in the pipeline layout for
// compatibility but intentionally undeclared here: bladed grass is fully
// procedural and samples no textures.

layout(set = 2, binding = 0) uniform WindParamsUBO {
    vec4 windDirAndStrength;
    vec4 windNoise;
    vec4 windShape;
    vec4 windTurbulence;
    vec4 densityParams;
    vec4 cameraPosAndFalloff;
} windParams;


layout(push_constant) uniform PushConstants {
    float billboardScale;
    float windEnabled;
    float windTime;
    float impostorDistance;
};

vec3 fragPosWorld; // set in main() — required by shadows.glsl cascades 1 & 2

#include "../../includes/shadow/Shadows.glsl"
#include "../../includes/vegetation/GrassShading.glsl"

void main() {
    // Debug: raw LOD source, no shading. Red = bladed grass (this shader),
    // blue = impostor (impostors.frag). The color edge is the hand-off.
    if (ubo.debugMode == DEBUG_MODE_VEGETATION_LOD) {
        outColor = vec4(1.0, 0.0, 0.0, 1.0);
        return;
    }

    bool shadowPass = windEnabled < 0.0;

    float heightT = clamp(inTexCoord.y, 0.0, 1.0);
    float side = inTexCoord.x;
    int variant = inBrushIndex;

    vec3 rootColor, tipColor;
    bladePalette(variant, rootColor, tipColor);

    // Per-tuft variation (shared formula with the impostor path, which
    // evaluates it at the depth-reconstructed blade pos).
    float varMix = grassVariation(inWorldPos);
    rootColor *= (1.0 + varMix);
    tipColor *= (1.0 + varMix);

    // Root->tip gradient with a slight edge darken across the blade width
    // (fake fold shading: center brighter, edges darker).
    vec3 albedo = mix(rootColor, tipColor, pow(heightT, 1.25));
    albedo *= 1.0 - 0.18 * clamp(abs(side), 0.0, 1.0);

    // Blade world normal: face normal flipped for back-faces (two-sided).
    vec3 faceN = gl_FrontFacing ? normalize(inPlaneNormal) : -normalize(inPlaneNormal);

    outColor = vec4(grassShade(albedo, faceN, inLightDirN, inWorldPos, shadowPass), 1.0);
}

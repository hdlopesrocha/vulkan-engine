#version 450

#include "../../ubo/UniformObject.glsl"
#include "../../ubo/WindParamsUBO.glsl"

#include "../../includes/Locations.glsl"

layout(location = ATTR_UV) in vec2 inCornerUV;
layout(location = ATTR_INSTANCE) in vec4 instanceData; // xyz=world pos, w=billboard index + rotFrac
layout(location = ATTR_VEG_NORMAL) in vec3 inInstanceNormal; // surface normal: re-anchor quad onto it
// Baked height scale (binding 2, perf report 22 C2/H4); see VegetationRenderer.vert.
layout(location = ATTR_VEG_AUX) in float inBakedHeight;

layout(location = VARY_UV) out vec3 outTexCoord;
layout(location = VARY_POSWORLD) out vec3 outWorldPos;
// xyz = instance base position, w = runtime height scale (hs). The fragment
// reconstruction scales the captured surface point by the SAME hs the quad
// geometry uses, so the written depth tracks the rasterized tuft exactly.
layout(location = VARY_POSLIGHT) flat out vec4 outInstanceOffset;
// Instance rotation + terrain surface normal: the reconstruction applies the
// same rotateY(hs * p) + tilt-to-surface transform the blade VS bakes in.
layout(location = VARY_ROTFRAC) flat out vec2 outCosSinT;
layout(location = VARY_PLANE_NORMAL) flat out vec3 outSurfN;

// Canonical scene UBO (set=0 binding=0): the shared struct IS the block
// layout — no packed/named-view split.
layout(std140, set = 0, binding = 0) uniform SolidParamsBlock {
    UniformObject ubo;
};

// Canonical wind params block (set=2 binding=0): the shared struct IS the
// block layout — no packed/named-view split.
layout(std140, set = 2, binding = 0) uniform WindParamsBlock {
    WindParamsUBO windParams;
};

layout(push_constant) uniform PushConstants {
    float billboardScale;
    float windEnabled;
    float windTime;
    float impostorDistance;
};

#include "../../includes/noise/Perlin2D.glsl"
#include "../../includes/vegetation/VegetationCommon.glsl"
#include "../../includes/vegetation/ImpostorViews.glsl"

void main() {
    // Sentinel: instance was skipped by generator (empty biome or steep slope).
    if (instanceData.w < 0.0) {
        outTexCoord = vec3(0.0); outInstanceOffset = vec4(0.0);
        outWorldPos = vec3(0.0); outSurfN = vec3(0.0, 1.0, 0.0); outCosSinT = vec2(1.0, 0.0);
        gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec3 worldPos = instanceData.xyz;
    int billboardIdx = int(floor(instanceData.w));
    float rotFrac = fract(instanceData.w);

    if (impostorDistance <= 0.0) {
        outTexCoord = vec3(0.0); outInstanceOffset = vec4(worldPos, 1.0);
        outSurfN = vec3(0.0, 1.0, 0.0); outCosSinT = vec2(1.0, 0.0);
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    float mainCamDist = distance(windParams.cameraPosition, worldPos);
    // Direct hand-off: the billboard culls at impostorDistance (VegetationRenderer.vert),
    // so the impostor takes over at exactly the same per-instance distance.
    if (mainCamDist < impostorDistance) {
        outTexCoord = vec3(0.0); outInstanceOffset = vec4(worldPos, 1.0);
        outSurfN = vec3(0.0, 1.0, 0.0); outCosSinT = vec2(1.0, 0.0);
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    float densityFactor = densityFactorForDistance(mainCamDist);
    if (densityFactor < 0.9999) {
        float keep = perlin2d_hash13(vec3(worldPos.xz * 0.03125, float(billboardIdx) + worldPos.y * 0.0078125));
        if (keep > densityFactor) {
            outTexCoord = vec3(0.0); outInstanceOffset = vec4(worldPos, 1.0);
            outSurfN = vec3(0.0, 1.0, 0.0); outCosSinT = vec2(1.0, 0.0);
            gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
            return;
        }
    }

    // Instance rotation (same convention as the blade VS and the impostor
    // color VS: theta = rotFrac * 2pi).
    float theta = rotFrac * 6.28318530718;
    outCosSinT = vec2(cos(theta), sin(theta));

    // Best-view select (perf report 26 H4): shared helper, bit-identical
    // layer to the inline 20-iteration loop it replaces (see
    // VegetationRendererImpostor.vert note on why no cross-vertex broadcast).
#ifdef VEG_SHADOW_PASS
    // Shadow pass (VegetationRendererImpostorShadow.vert.spv): the rasterized
    // quad faces the LIGHT, and the capture layer is the view nearest the
    // light. Facing the main camera here (the color/depth orientation) makes
    // the quad grazing to the sun, collapsing its shadow-map footprint to a
    // sliver, and biases the reconstructed-depth silhouette by up to half a
    // Fibonacci view. ubo.lightDirection points from the light toward the
    // scene, so -lightDirection is the direction toward the light.
    vec3 toCamera = normalize(-ubo.lightDirection);
#else
    vec3 toCamera = normalize(ubo.viewPosition - worldPos);
#endif

    int layerIdx = impostorBestLayerIndex(toCamera, rotFrac, billboardIdx);

    float hs = inBakedHeight;

    // M5 (perf report 28): shared placement core (center -> quad -> offset ->
    // finalPos -> UV crop); the prologue above stays twin-specific.
    #include "../../includes/vegetation/ImpostorPlacement.glsl"
    outInstanceOffset = vec4(worldPos, hs);
    outSurfN = surfN;
}

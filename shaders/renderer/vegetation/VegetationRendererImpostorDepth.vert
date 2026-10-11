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
layout(location = VARY_TANGENTWS) flat out vec3 outInstanceOffset;

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
        outTexCoord = vec3(0.0); outInstanceOffset = vec3(0.0);
        outWorldPos = vec3(0.0);
        gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    vec3 worldPos = instanceData.xyz;
    int billboardIdx = int(floor(instanceData.w));
    float rotFrac = fract(instanceData.w);

    if (impostorDistance <= 0.0) {
        outTexCoord = vec3(0.0); outInstanceOffset = worldPos;
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    float mainCamDist = distance(windParams.cameraPosition, worldPos);
    // Direct hand-off: the billboard culls at impostorDistance (VegetationRenderer.vert),
    // so the impostor takes over at exactly the same per-instance distance.
    if (mainCamDist < impostorDistance) {
        outTexCoord = vec3(0.0); outInstanceOffset = worldPos;
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    float densityFactor = densityFactorForDistance(mainCamDist);
    if (densityFactor < 0.9999) {
        float keep = perlin2d_hash13(vec3(worldPos.xz * 0.03125, float(billboardIdx) + worldPos.y * 0.0078125));
        if (keep > densityFactor) {
            outTexCoord = vec3(0.0); outInstanceOffset = worldPos;
            gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
            return;
        }
    }

    outInstanceOffset = worldPos;

    // Best-view select (perf report 26 H4): shared helper, bit-identical
    // layer to the inline 20-iteration loop it replaces (see
    // VegetationRendererImpostor.vert note on why no cross-vertex broadcast).
    vec3 toCamera = normalize(ubo.viewPosition - worldPos);

    int layerIdx = impostorBestLayerIndex(toCamera, rotFrac, billboardIdx);

    float hs = inBakedHeight;

    // M5 (perf report 28): shared placement core (center -> quad -> offset ->
    // finalPos -> UV crop); the prologue above stays twin-specific.
    #include "../../includes/vegetation/ImpostorPlacement.glsl"
    outInstanceOffset = worldPos;
}

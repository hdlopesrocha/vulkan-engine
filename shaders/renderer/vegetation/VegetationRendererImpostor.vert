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
// H8 (perf report 27): per-instance trig as a flat varying (replaces the
// flat rotFrac, whose only fragment consumer was the per-pixel theta/cosT/
// sinT rebuild). Same location, no new interpolators.
layout(location = VARY_ROTFRAC) flat out vec2 outCosSinT;
// H8 (perf report 27): uniform light direction normalized once per vertex,
// not per pixel (uniform input, identical value). VARY_TANGENTWS is unused
// in the impostor color interface.
layout(location = VARY_TANGENTWS) flat out vec3 outLightDirN;
layout(location = VARY_POSLIGHT) flat out vec3 outInstanceOffset;

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
    vec3 worldPos = instanceData.xyz;
    int billboardIdx = int(floor(instanceData.w));
    float rotFrac = fract(instanceData.w);

    vec3 camPos = ubo.viewPosition;
    float dist = distance(worldPos, camPos);

    // Direct hand-off, no dithered cross-fade (canonical note in
    // VegetationRenderer.vert, perf report 26 L15).
    if (impostorDistance <= 0.0 || dist < impostorDistance) {
        outTexCoord = vec3(0.0); outWorldPos = worldPos;
        outInstanceOffset = worldPos;
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    float densityFactor = densityFactorForDistance(dist);
    if (densityFactor < 0.9999) {
        float keep = perlin2d_hash13(vec3(worldPos.xz * 0.03125, float(billboardIdx) + worldPos.y * 0.0078125));
        if (keep > densityFactor) {
            outTexCoord = vec3(0.0); outWorldPos = worldPos;
            outInstanceOffset = worldPos;
            gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
            return;
        }
    }

    outInstanceOffset = worldPos;

    // Best-view select (perf report 26 H4): the winning view depends only on
    // instance rotation + camera direction, so all 4 quad corners agree; the
    // shared helper in ImpostorViews.glsl runs the same 20-iteration loop and
    // returns a bit-identical layer. A true once-per-quad broadcast is NOT
    // done here: vertex invocations cannot share values without subgroup ops,
    // and quad-in-one-subgroup is not portable.
    vec3 toCamera = normalize(camPos - worldPos);

    // H8 (perf report 27): per-instance trig once per vertex, not once per
    // pixel. theta/cosT/sinT are a pure function of the flat rotFrac (same
    // verbatim expression the fragment stage evaluated), so every pixel of
    // the instance receives the identical pair via the flat varying; the
    // per-pixel rotate/normalize/lighting chain in the fragment stage is
    // untouched. The uniform light-direction normalize moves here for the
    // same reason (uniform input, identical value, one evaluation per
    // vertex instead of per pixel).
    float theta = rotFrac * 6.28318530718;
    outCosSinT = vec2(cos(theta), sin(theta));
    outLightDirN = normalize(-ubo.lightDirection);

    int layerIdx = impostorBestLayerIndex(toCamera, rotFrac, billboardIdx);

    float hs = inBakedHeight;

    // M5 (perf report 28): shared placement core (center -> quad -> offset ->
    // finalPos -> UV crop); the prologue above stays twin-specific.
    #include "../../includes/vegetation/ImpostorPlacement.glsl"
}

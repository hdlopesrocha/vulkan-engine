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
layout(location = VARY_FACE_NORMAL) flat out vec3 outFaceNormal;
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
        outTexCoord = vec3(0.0); outWorldPos = worldPos; outFaceNormal = vec3(0.0, 1.0, 0.0);
        outInstanceOffset = worldPos;
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    float densityFactor = densityFactorForDistance(dist);
    if (densityFactor < 0.9999) {
        float keep = perlin2d_hash13(vec3(worldPos.xz * 0.03125, float(billboardIdx) + worldPos.y * 0.0078125));
        if (keep > densityFactor) {
            outTexCoord = vec3(0.0); outWorldPos = worldPos; outFaceNormal = vec3(0.0, 1.0, 0.0);
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

    vec3 center;
    float quadHalfW;
    float quadHalfH;

    // Match the capture setup: the plant was captured at heightScale=1.0 with a
    // fixed-size square framebuffer. The plant occupies only 34.6% of the image
    // height (top/bottom 32.7% are clear). The quad and its centre scale
    // uniformly with the runtime height scale (hs), exactly like the billboard
    // (VegetationRenderer.vert scales its corners by billboardScale * hs), so the
    // impostor is the same size as the plant it replaces. The UV crop maps the
    // captured bbox (a fixed fraction of the image) to the whole quad, so the
    // texture scales with the quad; it must NOT scale with hs itself.
    // The quad stays glued to sloped ground by rising along the surface normal.
    vec3 surfN = inInstanceNormal;
    float surfNLen2 = dot(surfN, surfN);
    surfN = (surfNLen2 > 1e-8) ? surfN * inversesqrt(surfNLen2) : vec3(0.0, 1.0, 0.0);
    center = worldPos + surfN * (billboardScale * hs * 0.5);
    vec3 worldUp = vec3(0.0, 1.0, 0.0);

    vec3 right;
    float sinElev = length(vec2(toCamera.x, toCamera.z));
    if (sinElev > 0.001) {
        right = normalize(cross(worldUp, toCamera));
    } else {
        right = vec3(1.0, 0.0, 0.0);
    }
    vec3 upDir = normalize(cross(toCamera, right));

    // Quad size: the captured plant bbox in world units, scaled by hs.
    // (The old code scaled the width by hs but left the height at
    // billboardScale and scaled uFrac by hs, which cancelled out and rendered
    // every impostor at the capture size - tall plants shrank and short ones
    // were cropped at the hand-off.)
    quadHalfW = 0.75 * billboardScale * hs;
    quadHalfH = 0.5  * billboardScale * hs;
    right = right * quadHalfW;
    vec3 up = upDir * quadHalfH;

    vec3 faceNorm = toCamera;
    outFaceNormal = faceNorm;

    float u = inCornerUV.x;
    float v = inCornerUV.y;
    vec3 offset = (u - 0.5) * 2.0 * right + (0.5 - v) * 2.0 * up;
    vec3 finalPos = center + offset;

    // Crop UV to the plant's bounding box within the captured image.
    // The plant occupies UV.V in [0.327, 0.673] (vertical) and
    // UV.U in [0.5 ± 1.5/2.8867] (horizontal), independent of hs.
    float uFrac = 1.5 / 2.886751346;      // 1.5 / (2 * 2.5 * tan(30°))
    float vFrac = 1.0 / 2.886751346;      // billboardScale / (2 * 2.5 * billboardScale * tan(30°))
    float vOff  = 0.5 - 0.5 / 2.886751346;
    outTexCoord = vec3(0.5 + (inCornerUV.x - 0.5) * uFrac,
                       inCornerUV.y * vFrac + vOff,
                       float(layerIdx));
    outWorldPos = finalPos;
    gl_Position = ubo.viewProjection * vec4(finalPos, 1.0);
}

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

    // Uniform hs scaling, matching impostors.vert: quad, centre and UV crop
    // must render the plant at the same size the billboard had.
    // The quad stays glued to sloped ground by rising along the surface normal.
    vec3 surfN = inInstanceNormal;
    float surfNLen2 = dot(surfN, surfN);
    surfN = (surfNLen2 > 1e-8) ? surfN * inversesqrt(surfNLen2) : vec3(0.0, 1.0, 0.0);
    vec3 center = worldPos + surfN * (billboardScale * hs * 0.5);
    vec3 worldUp = vec3(0.0, 1.0, 0.0);

    vec3 right;
    float sinElev = length(vec2(toCamera.x, toCamera.z));
    if (sinElev > 0.001) {
        right = normalize(cross(worldUp, toCamera));
    } else {
        right = vec3(1.0, 0.0, 0.0);
    }
    vec3 upDir = normalize(cross(toCamera, right));

    float quadHalfW = 0.75 * billboardScale * hs;
    float quadHalfH = 0.5  * billboardScale * hs;
    right = right * quadHalfW;
    vec3 up = upDir * quadHalfH;

    float u = inCornerUV.x;
    float v = inCornerUV.y;
    vec3 offset = (u - 0.5) * 2.0 * right + (0.5 - v) * 2.0 * up;
    vec3 finalPos = center + offset;

    float uFrac = 1.5 / 2.886751346;
    float vFrac = 1.0 / 2.886751346;
    float vOff  = 0.5 - 0.5 / 2.886751346;
    outTexCoord = vec3(0.5 + (inCornerUV.x - 0.5) * uFrac,
                       inCornerUV.y * vFrac + vOff,
                       float(layerIdx));
    outWorldPos = finalPos;
    outInstanceOffset = worldPos;
    gl_Position = ubo.viewProjection * vec4(finalPos, 1.0);
}

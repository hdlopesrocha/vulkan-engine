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
layout(location = VARY_ROTFRAC) flat out float outRotFrac;
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

void main() {
    vec3 worldPos = instanceData.xyz;
    int billboardIdx = int(floor(instanceData.w));
    float rotFrac = fract(instanceData.w);
    outRotFrac = rotFrac;

    vec3 camPos = ubo.viewPosition;
    float dist = distance(worldPos, camPos);

    // Direct hand-off: billboards own [0, impostorDistance), impostors own
    // [impostorDistance, ...). No dithered cross-fade (see VegetationRenderer.frag):
    // the fade dissolved the grass before the impostors read as vegetation.
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

    const float goldenAngle = 3.14159265358979323846 * (3.0 - 2.2360679774997896);
    const int NUM_VIEWS = 20;

    vec3 toCamera = normalize(camPos - worldPos);

    float instTheta = rotFrac * 6.28318530718;
    float cI = cos(instTheta);
    float sI = sin(instTheta);
    vec3 toCamera_canonical = vec3(
         cI * toCamera.x + sI * toCamera.z,
        toCamera.y,
        -sI * toCamera.x + cI * toCamera.z
    );

    int bestIdx = 0;
    float bestDot = -2.0;
    for (int i = 0; i < NUM_VIEWS; i++) {
        float y = 1.0 - (float(i) + 0.5) / float(NUM_VIEWS) * 2.0;
        float r = sqrt(max(0.0, 1.0 - y * y));
        float theta = goldenAngle * float(i);
        vec3 d = vec3(cos(theta) * r, y, sin(theta) * r);
        float dt = dot(toCamera_canonical, d);
        if (dt > bestDot) { bestDot = dt; bestIdx = i; }
    }

    int layerIdx = clamp(billboardIdx, 0, 2) * NUM_VIEWS + bestIdx;

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

#version 450

#include "includes/locations.glsl"

layout(location = ATTR_UV) in vec2 inCornerUV;
layout(location = ATTR_INSTANCE) in vec4 instanceData; // xyz=world pos, w=billboard index + rotFrac

layout(location = VARY_UV) out vec3 outTexCoord;
layout(location = VARY_POSWORLD) out vec3 outWorldPos;
layout(location = VARY_FACE_NORMAL) flat out vec3 outFaceNormal;
layout(location = VARY_ROTFRAC) flat out float outRotFrac;
layout(location = VARY_POSLIGHT) flat out vec3 outInstanceOffset;

layout(set = 0, binding = 0) uniform SolidParamsUBO {
    mat4 viewProjection;
    vec4 viewPos;
} ubo;

layout(push_constant) uniform PushConstants {
    float billboardScale;
    float windEnabled;
    float windTime;
    float impostorDistance;
    vec4 windDirAndStrength;
    vec4 windNoise;
    vec4 windShape;
    vec4 windTurbulence;
    vec4 densityParams;
    vec4 cameraPosAndFalloff;
};

#include "includes/perlin2d.glsl"
#include "includes/vegetation_common.glsl"

void main() {
    vec3 worldPos = instanceData.xyz;
    int billboardIdx = decodeBillboardIndex(instanceData.w);
    float rotFrac = decodeRotFrac(instanceData.w);
    outRotFrac = rotFrac;

    // Sentinel: empty-biome instances have w < 0
    if (instanceData.w < 0.0) {
        outTexCoord = vec3(0.0); outWorldPos = worldPos; outFaceNormal = vec3(0.0, 1.0, 0.0);
        outInstanceOffset = worldPos;
        gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
        return;
    }

    vec3 camPos = ubo.viewPos.xyz;
    float dist = distance(worldPos, camPos);

    if (impostorDistance <= 0.0 || dist < impostorDistance * 0.50) {
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

    float hs = decodeHeightScale(instanceData.w);

    // Match the capture setup: the plant was captured at heightScale=1.0 with a
    // fixed-size square framebuffer. The plant occupies only 34.6% of the image
    // height (top/bottom 32.7% are clear).  Adjust the quad size and UV mapping
    // to crop the empty margins and align the captured plant to the instance.
    vec3 center = worldPos + vec3(0.0, billboardScale * 0.5, 0.0);
    vec3 worldUp = vec3(0.0, 1.0, 0.0);

    vec3 right;
    float sinElev = length(vec2(toCamera.x, toCamera.z));
    if (sinElev > 0.001) {
        right = normalize(cross(worldUp, toCamera));
    } else {
        right = vec3(1.0, 0.0, 0.0);
    }
    vec3 upDir = normalize(cross(toCamera, right));

    // Quad size: width matches the plant's horizontal extent (1.5 × billboardScale × hs),
    // height matches the captured plant height (billboardScale, not hs).
    float quadHalfW = 0.75 * billboardScale * hs;
    float quadHalfH = 0.5  * billboardScale;
    right = right * quadHalfW;
    vec3 up = upDir * quadHalfH;

    vec3 faceNorm = toCamera;
    outFaceNormal = faceNorm;

    float u = inCornerUV.x;
    float v = inCornerUV.y;
    vec3 offset = (u - 0.5) * 2.0 * right + (0.5 - v) * 2.0 * up;
    vec3 finalPos = center + offset;

    // Crop UV to the plant's bounding box within the captured image.
    // The plant occupies UV.V in [0.327, 0.673] (vertical) and UV.U projects
    // via the instance's world-space position × hs (horizontal).
    float uFrac = hs * 1.5 / (2.886751346); // hs * 1.5 / (2 * 2.5 * tan(30°))
    float vFrac = 1.0 / (2.886751346);      // billboardScale / (2 * 2.5 * billboardScale * tan(30°))
    float vOff  = 0.5 - 0.5 / (2.886751346);
    outTexCoord = vec3(0.5 + (inCornerUV.x - 0.5) * uFrac,
                       inCornerUV.y * vFrac + vOff,
                       float(layerIdx));
    outWorldPos = finalPos;
    gl_Position = ubo.viewProjection * vec4(finalPos, 1.0);
}

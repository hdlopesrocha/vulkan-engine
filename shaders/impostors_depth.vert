#version 450

#include "includes/locations.glsl"

layout(location = ATTR_UV) in vec2 inCornerUV;
layout(location = ATTR_INSTANCE) in vec4 instanceData; // xyz=world pos, w=billboard index + rotFrac

layout(location = VARY_UV) out vec3 outTexCoord;
layout(location = VARY_POSWORLD) out vec3 outWorldPos;
layout(location = VARY_TANGENTWS) flat out vec3 outInstanceOffset;

layout(set = 0, binding = 0) uniform SolidParamsUBO {
    mat4 viewProjection;
    vec4 viewPos;
} ubo;

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

#include "includes/perlin2d.glsl"
#include "includes/vegetation_common.glsl"

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

    float mainCamDist = distance(windParams.cameraPosAndFalloff.xyz, worldPos);
    if (mainCamDist < impostorDistance * 0.50) {
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

    const float goldenAngle = 3.14159265358979323846 * (3.0 - 2.2360679774997896);
    const int NUM_VIEWS = 20;

    vec3 toCamera = normalize(ubo.viewPos.xyz - worldPos);

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

    float hs = vegetationHeightScale(worldPos.xz);

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

    float quadHalfW = 0.75 * billboardScale * hs;
    float quadHalfH = 0.5  * billboardScale;
    right = right * quadHalfW;
    vec3 up = upDir * quadHalfH;

    float u = inCornerUV.x;
    float v = inCornerUV.y;
    vec3 offset = (u - 0.5) * 2.0 * right + (0.5 - v) * 2.0 * up;
    vec3 finalPos = center + offset;

    float uFrac = hs * 1.5 / (2.886751346);
    float vFrac = 1.0 / (2.886751346);
    float vOff  = 0.5 - 0.5 / (2.886751346);
    outTexCoord = vec3(0.5 + (inCornerUV.x - 0.5) * uFrac,
                       inCornerUV.y * vFrac + vOff,
                       float(layerIdx));
    outWorldPos = finalPos;
    outInstanceOffset = worldPos;
    gl_Position = ubo.viewProjection * vec4(finalPos, 1.0);
}

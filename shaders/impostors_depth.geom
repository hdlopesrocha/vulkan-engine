#version 450

#include "includes/locations.glsl"

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

layout(location = VARY_POSWORLD) in vec3 inWorldPos[];
layout(location = VARY_BRUSHPATCH) flat in int inBillboardIndex[];
layout(location = VARY_ROTFRAC) in float inRotFrac[];

layout(location = VARY_UV) out vec3 outTexCoord;       // xy=UV, z=float(layerIdx)
layout(location = VARY_TANGENTWS) flat out vec3 outInstanceOffset;

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
    vec3 worldPos = inWorldPos[0];
    int  billboardIdx = inBillboardIndex[0];

    if (impostorDistance <= 0.0) return;

    float mainCamDist = distance(cameraPosAndFalloff.xyz, worldPos);
    if (mainCamDist < impostorDistance * 0.85) return;

    float densityFactor = densityFactorForDistance(mainCamDist);
    if (densityFactor < 0.9999) {
        float keep = perlin2d_hash13(vec3(worldPos.xz * 0.03125, float(billboardIdx) + worldPos.y * 0.0078125));
        if (keep > densityFactor) return;
    }

    // ── Fibonacci view selection ──────────────────────────────────────────
    const float goldenAngle = 3.14159265358979323846 * (3.0 - 2.2360679774997896);
    const int   NUM_VIEWS   = 20;

    vec3 toCamera = normalize(ubo.viewPos.xyz - worldPos);

    float instTheta = inRotFrac[0] * 6.28318530718;
    float cI = cos(instTheta);
    float sI = sin(instTheta);
    vec3 toCamera_canonical = vec3(
         cI * toCamera.x + sI * toCamera.z,
        toCamera.y,
        -sI * toCamera.x + cI * toCamera.z
    );

    int   bestIdx = 0;
    float bestDot = -2.0;
    for (int i = 0; i < NUM_VIEWS; i++) {
        float y     = 1.0 - (float(i) + 0.5) / float(NUM_VIEWS) * 2.0;
        float r     = sqrt(max(0.0, 1.0 - y * y));
        float theta = goldenAngle * float(i);
        vec3  d     = vec3(cos(theta) * r, y, sin(theta) * r);
        float dt    = dot(toCamera_canonical, d);
        if (dt > bestDot) { bestDot = dt; bestIdx = i; }
    }

    int layerIdx = clamp(billboardIdx, 0, 2) * NUM_VIEWS + bestIdx;

    float scaledBillboard = billboardScale * vegetationHeightScale(worldPos.xz);

    // ── Camera-facing quad ────────────────────────────────────────────────
    vec3 center  = worldPos + vec3(0.0, scaledBillboard * 0.5, 0.0);
    vec3 worldUp = vec3(0.0, 1.0, 0.0);

    vec3 right;
    float sinElev = length(vec2(toCamera.x, toCamera.z));
    if (sinElev > 0.001) {
        right = normalize(cross(worldUp, toCamera));
    } else {
        right = vec3(1.0, 0.0, 0.0);
    }
    vec3 upDir = normalize(cross(toCamera, right));

    const float kHalf = 1.44338;
    right = right * (scaledBillboard * kHalf);
    vec3 up = upDir * (scaledBillboard * kHalf);

    float layer = float(layerIdx);

    vec3 vBL = center - right - up;
    vec3 vBR = center + right - up;
    vec3 vTL = center - right + up;
    vec3 vTR = center + right + up;

    outInstanceOffset = worldPos;

    gl_Position = ubo.viewProjection * vec4(vBL, 1.0); outTexCoord = vec3(0.0, 1.0, layer); EmitVertex();
    gl_Position = ubo.viewProjection * vec4(vBR, 1.0); outTexCoord = vec3(1.0, 1.0, layer); EmitVertex();
    gl_Position = ubo.viewProjection * vec4(vTL, 1.0); outTexCoord = vec3(0.0, 0.0, layer); EmitVertex();
    gl_Position = ubo.viewProjection * vec4(vTR, 1.0); outTexCoord = vec3(1.0, 0.0, layer); EmitVertex();

    EndPrimitive();
}

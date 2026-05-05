#version 450
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

layout(location = 0) in vec3 inWorldPos[];
layout(location = 1) flat in int inBillboardIndex[];
layout(location = 2) in float inRotFrac[];

layout(location = 0) out vec3 outTexCoord;    // xy=UV, z=float(layerIdx)
layout(location = 1) out vec3 outWorldPos;    // world-space vertex position (for specular)
layout(location = 2) flat out vec3 outFaceNormal; // world-space face normal (toward camera in XZ)
layout(location = 3) flat out float outRotFrac;   // per-instance Y-rotation fraction [0,1)

// Must match SolidParamsUBO — only read the first two fields.
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

    vec3 camPos = ubo.viewPos.xyz;
    float dist  = distance(worldPos, camPos);

    // Skip if feature disabled OR instance is within near-vegetation distance
    // (the regular vegetation pipeline handles those instances).
    // Start 15% before impostorDistance to overlap with the vegetation cross-fade zone.
    if (impostorDistance <= 0.0 || dist < impostorDistance * 0.85) return;

    // Distance-based density thinning — same stochastic model as vegetation.geom.
    float densityFactor = densityFactorForDistance(dist);
    if (densityFactor < 0.9999) {
        float keep = perlin2d_hash13(vec3(worldPos.xz * 0.03125, float(billboardIdx) + worldPos.y * 0.0078125));
        if (keep > densityFactor) return;
    }

    // ── Fibonacci sphere view selection ────────────────────────────────────
    // Directions are computed from the golden-angle formula to match what
    // ImpostorCapture uses on the CPU side.  viewDirs[i] = FROM center TO camera.
    // We compare against toCamera = normalize(camPos - worldPos).
    const float goldenAngle = 3.14159265358979323846 * (3.0 - 2.2360679774997896);
    const int   NUM_VIEWS   = 20;

    vec3 toCamera = normalize(camPos - worldPos);

    // ── Per-instance Y-rotation ──────────────────────────────────────────────
    // The capture was done at θ=0.  To find the correct Fibonacci layer we must
    // express toCamera in the canonical (θ=0) frame of the plant, i.e. rotate
    // it backward by the instance's Y rotation.
    float instTheta = inRotFrac[0] * 6.28318530718;
    float cI = cos(instTheta);
    float sI = sin(instTheta);
    // rotateY(toCamera, -instTheta)
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

    // Layer = billboardType * 20 + viewIdx  (matches ImpostorCapture layout).
    int layerIdx = clamp(billboardIdx, 0, 2) * NUM_VIEWS + bestIdx;

    // Per-instance height variation matching vegetation.geom (same noise function + scale).
    float scaledBillboard = billboardScale * vegetationHeightScale(worldPos.xz);

    // ── Spherical billboard quad ──────────────────────────────────────────
    // Center between ground level and top of vegetation.
    vec3 center  = worldPos + vec3(0.0, scaledBillboard * 0.5, 0.0);
    vec3 worldUp = vec3(0.0, 1.0, 0.0);

    // right is perpendicular to both worldUp and the full 3-D toCamera vector.
    // When the camera is directly above/below, fall back to an arbitrary right axis.
    vec3 right;
    float sinElev = length(vec2(toCamera.x, toCamera.z));
    if (sinElev > 0.001) {
        right = normalize(cross(worldUp, toCamera));
    } else {
        right = vec3(1.0, 0.0, 0.0);
    }
    // up is perpendicular to both the view vector and right — ensures the quad
    // is always exactly perpendicular to the view vector.
    vec3 upDir = normalize(cross(toCamera, right));

    // The capture camera uses captureDist = billboardScale * 2.5 and FOV = 60°.
    // The full texture maps to frustum_half_size = captureDist * tan(30°) = billboardScale * 1.44338.
    // The quad must cover that same world-space extent so UVs map 1:1 to the captured image.
    const float kHalf = 1.44338; // = 2.5 * tan(30°)
    right = right * (scaledBillboard * kHalf);
    vec3 up = upDir * (scaledBillboard * kHalf);

    float layer = float(layerIdx);
    // Face normal is the full 3-D direction toward the camera.
    vec3 faceNorm = toCamera;

    // Emit quad (triangle strip BL → BR → TL → TR).
    vec3 vBL = center - right - up;
    vec3 vBR = center + right - up;
    vec3 vTL = center - right + up;
    vec3 vTR = center + right + up;

    outRotFrac = inRotFrac[0];

    gl_Position = ubo.viewProjection * vec4(vBL, 1.0);
    outTexCoord = vec3(0.0, 1.0, layer); outWorldPos = vBL; outFaceNormal = faceNorm; EmitVertex();

    gl_Position = ubo.viewProjection * vec4(vBR, 1.0);
    outTexCoord = vec3(1.0, 1.0, layer); outWorldPos = vBR; outFaceNormal = faceNorm; EmitVertex();

    gl_Position = ubo.viewProjection * vec4(vTL, 1.0);
    outTexCoord = vec3(0.0, 0.0, layer); outWorldPos = vTL; outFaceNormal = faceNorm; EmitVertex();

    gl_Position = ubo.viewProjection * vec4(vTR, 1.0);
    outTexCoord = vec3(1.0, 0.0, layer); outWorldPos = vTR; outFaceNormal = faceNorm; EmitVertex();

    EndPrimitive();
}

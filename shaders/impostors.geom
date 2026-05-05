#version 450
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

layout(location = 0) in vec3 inWorldPos[];
layout(location = 1) flat in int inBillboardIndex[];

layout(location = 0) out vec3 outTexCoord;    // xy=UV, z=float(layerIdx)
layout(location = 1) out vec3 outWorldPos;    // world-space vertex position (for specular)
layout(location = 2) flat out vec3 outFaceNormal; // world-space face normal (toward camera in XZ)

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

float hash13(vec3 p3) {
    p3 = fract(p3 * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float densityFactorForDistance(float distanceToCamera) {
    if (densityParams.x < 0.5) return 1.0;
    float nearDistance = max(0.0, densityParams.y);
    float minFactor = clamp(densityParams.w, 0.0, 1.0);
    float falloff = cameraPosAndFalloff.w;
    if (distanceToCamera <= nearDistance || minFactor >= 1.0 || falloff <= 0.0) {
        return 1.0;
    }
    float density = exp(-falloff * (distanceToCamera - nearDistance));
    return clamp(density, minFactor, 1.0);
}

void main() {
    vec3 worldPos = inWorldPos[0];
    int  billboardIdx = inBillboardIndex[0];

    vec3 camPos = ubo.viewPos.xyz;
    float dist  = distance(worldPos, camPos);

    // Skip if feature disabled OR instance is within near-vegetation distance
    // (the regular vegetation pipeline handles those instances).
    if (impostorDistance <= 0.0 || dist < impostorDistance) return;

    // Distance-based density thinning — same stochastic model as vegetation.geom.
    float densityFactor = densityFactorForDistance(dist);
    if (densityFactor < 0.9999) {
        float keep = hash13(vec3(worldPos.xz * 0.03125, float(billboardIdx) + worldPos.y * 0.0078125));
        if (keep > densityFactor) return;
    }

    // ── Fibonacci sphere view selection ────────────────────────────────────
    // Directions are computed from the golden-angle formula to match what
    // ImpostorCapture uses on the CPU side.  viewDirs[i] = FROM center TO camera.
    // We compare against toCamera = normalize(camPos - worldPos).
    const float goldenAngle = 3.14159265358979323846 * (3.0 - 2.2360679774997896);
    const int   NUM_VIEWS   = 20;

    vec3 toCamera = normalize(camPos - worldPos);

    int   bestIdx = 0;
    float bestDot = -2.0;
    for (int i = 0; i < NUM_VIEWS; i++) {
        float y     = 1.0 - (float(i) + 0.5) / float(NUM_VIEWS) * 2.0;
        float r     = sqrt(max(0.0, 1.0 - y * y));
        float theta = goldenAngle * float(i);
        vec3  d     = vec3(cos(theta) * r, y, sin(theta) * r);
        float dt    = dot(toCamera, d);
        if (dt > bestDot) { bestDot = dt; bestIdx = i; }
    }

    // Layer = billboardType * 20 + viewIdx  (matches ImpostorCapture layout).
    int layerIdx = clamp(billboardIdx, 0, 2) * NUM_VIEWS + bestIdx;

    // ── Cylindrical billboard quad ──────────────────────────────────────────
    // Center between ground level and top of vegetation.
    vec3 center  = worldPos + vec3(0.0, billboardScale * 0.5, 0.0);
    vec3 worldUp = vec3(0.0, 1.0, 0.0);

    // Right = perpendicular to both worldUp and the XZ-projected view dir.
    vec3 toCamXZ = vec3(toCamera.x, 0.0, toCamera.z);
    float xzLen  = length(toCamXZ);
    vec3 right;
    if (xzLen > 0.001) {
        toCamXZ = toCamXZ / xzLen;
        right   = normalize(cross(worldUp, toCamXZ));
    } else {
        // Camera directly above/below — pick arbitrary right axis.
        right = vec3(1.0, 0.0, 0.0);
    }
    // The capture camera uses captureDist = billboardScale * 2.5 and FOV = 60°.
    // The full texture maps to frustum_half_size = captureDist * tan(30°) = billboardScale * 1.44338.
    // The quad must cover that same world-space extent so UVs map 1:1 to the captured image.
    const float kHalf = 1.44338; // = 2.5 * tan(30°)
    right = right * (billboardScale * kHalf);
    vec3 up = worldUp * (billboardScale * kHalf);

    float layer = float(layerIdx);
    // Face normal is the XZ direction the billboard faces (toward camera).
    vec3 faceNorm = toCamXZ;

    // Emit quad (triangle strip BL → BR → TL → TR).
    vec3 vBL = center - right - up;
    vec3 vBR = center + right - up;
    vec3 vTL = center - right + up;
    vec3 vTR = center + right + up;

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

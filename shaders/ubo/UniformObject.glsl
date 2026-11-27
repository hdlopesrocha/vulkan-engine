#ifndef UNIFORM_OBJECT_GLSL
#define UNIFORM_OBJECT_GLSL

// Canonical scene UBO (std140, 288 B). Single definition shared with the CPU:
// vulkan/ubo/UniformObject.hpp — same names, fields and offsets.
// Offset table (verified against the C++ static_asserts):
//   viewProjection 0, invViewProjection 64, viewPosition 128,
//   lightElevation 140, lightDirection 144, cubemapCapture 156,
//   lightColor 160, triplanarThreshold 172, triplanarExponent 176,
//   normalMappingEnabled 180, shadowsEnabled 184, debugMode 188,
//   roughnessEnabled 192, ambientOcclusionEnabled 196, tessNearDist 200,
//   tessFarDist 204, tessellationFactor 208, tessellationEnabled 212,
//   nearPlane 216, farPlane 220, brushTextureIndex 224, brushMode 228,
//   brushPhase 232, brushHsv 240, raymarchScreenEps 252,
//   displacementEnabled 256, displacementAOEnabled 260,
//   displacementMaxBound 264, shadowSoftness 268,
//   shadowMaxDistance 272, shadowMaxSteps 276, shadowReserved 280.
// Block size 288.
// Flags are uint (compare with == 0u / != 0u); debugMode is int.
struct UniformObject {
    mat4 viewProjection;        // offset   0
    mat4 invViewProjection;     // offset  64
    vec3 viewPosition;          // offset 128
    float lightElevation;       // offset 140  lightDirection.y
    vec3 lightDirection;        // offset 144  FROM the light source toward the scene
    uint cubemapCapture;        // offset 156
    vec3 lightColor;            // offset 160
    float triplanarThreshold;   // offset 172
    float triplanarExponent;    // offset 176
    uint normalMappingEnabled;  // offset 180
    uint shadowsEnabled;        // offset 184
    int debugMode;              // offset 188
    uint roughnessEnabled;      // offset 192
    uint ambientOcclusionEnabled; // offset 196
    float tessNearDist;         // offset 200
    float tessFarDist;          // offset 204
    float tessellationFactor;   // offset 208
    uint tessellationEnabled;   // offset 212
    float nearPlane;            // offset 216
    float farPlane;             // offset 220
    uint brushTextureIndex;     // offset 224
    uint brushMode;             // offset 228
    float brushPhase;           // offset 232
    vec3 brushHsv;              // offset 240
    float raymarchScreenEps;    // offset 252  octree raymarch SS hit tolerance (px, 0 = off)
    uint displacementEnabled;      // offset 256  master gate for the new path
    uint displacementAOEnabled;    // offset 260  SDF ambient occlusion
    float displacementMaxBound;    // offset 264  max |h| over uploaded materials
    float shadowSoftness;          // offset 268  0 = hard, else penumbra k
    float shadowMaxDistance;       // offset 272  0 = far plane
    uint shadowMaxSteps;           // offset 276  shadow-ray step budget
    uint shadowReserved;           // offset 280
};

#endif // UNIFORM_OBJECT_GLSL

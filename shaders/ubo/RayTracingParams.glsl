#ifndef RAY_TRACING_PARAMS_GLSL
#define RAY_TRACING_PARAMS_GLSL

// Canonical ray-tracing parameters UBO (std140, 288 B). Single definition
// shared with the CPU: vulkan/renderer/RayTracingResources.hpp — same names,
// fields and offsets.
// Offset table (verified against the C++ static_asserts):
//   invViewProj 0, prevViewProj 64, viewPosition 128, maxReflectDistance 140,
//   sunDirection 144, maxRefractDistance 156, sunColor 160,
//   maxShadowDistance 172, absorptionColor 176, absorptionScale 188,
//   rtResolution 192, invRtResolution 200, roughnessThreshold 208,
//   waterIor 212, maxWaterThickness 216, coarseBoxSize 220,
//   maxReflectionBounces 224, debugMode 228, selfSkipDist 232,
//   reflectionContribMin 236, nearPlane 240, farPlane 244,
//   reflectionsEnabled 248, refractionsEnabled 252, thicknessEnabled 256,
//   localShadowsEnabled 260, tlasReady 264, useWaterPipeline 268,
//   checkerboardReflections 272, singleRay 276, waterReflections 280,
//   rayTracedWaterDepth 284.
// Flags are uint (compare with == 0u / != 0u).
struct RayTracingParams {
    mat4 invViewProj;            // offset   0
    mat4 prevViewProj;           // offset  64  previous frame's VP (temporal SSR)
    vec3 viewPosition;           // offset 128
    float maxReflectDistance;    // offset 140
    vec3 sunDirection;           // offset 144  xyz = direction TO sun
    float maxRefractDistance;    // offset 156
    vec3 sunColor;               // offset 160
    float maxShadowDistance;     // offset 172
    vec3 absorptionColor;        // offset 176  Beer-Lambert coeff
    float absorptionScale;       // offset 188
    vec2 rtResolution;           // offset 192  dispatch size
    vec2 invRtResolution;        // offset 200  1/size
    float roughnessThreshold;    // offset 208
    float waterIor;              // offset 212
    float maxWaterThickness;     // offset 216
    float coarseBoxSize;         // offset 220
    int maxReflectionBounces;    // offset 224
    int debugMode;               // offset 228
    float selfSkipDist;          // offset 232
    float reflectionContribMin;  // offset 236
    float nearPlane;             // offset 240
    float farPlane;              // offset 244
    uint reflectionsEnabled;     // offset 248
    uint refractionsEnabled;     // offset 252
    uint thicknessEnabled;       // offset 256
    uint localShadowsEnabled;    // offset 260
    uint tlasReady;              // offset 264
    uint useWaterPipeline;       // offset 268
    uint checkerboardReflections;// offset 272
    uint singleRay;              // offset 276
    uint waterReflections;       // offset 280
    uint rayTracedWaterDepth;    // offset 284
};

#endif // RAY_TRACING_PARAMS_GLSL

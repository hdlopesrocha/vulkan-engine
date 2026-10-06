#ifndef UNIFORM_OBJECT_GLSL
#define UNIFORM_OBJECT_GLSL

// Canonical scene UBO (std140, 448 B). Single definition shared with the CPU:
// vulkan/ubo/UniformObject.hpp — same names, fields and offsets.
// Offset table (verified against the C++ static_asserts):
//   viewProjection 0, invViewProjection 64, lightSpaceMatrix 128,
//   lightSpaceMatrix1 192, lightSpaceMatrix2 256, viewPosition 320,
//   lightElevation 332, lightDirection 336, cubemapCapture 348,
//   lightColor 352, triplanarThreshold 364, triplanarExponent 368,
//   normalMappingEnabled 372, shadowsEnabled 376, debugMode 380,
//   roughnessEnabled 384, ambientOcclusionEnabled 388, tessNearDist 392,
//   tessFarDist 396, tessellationFactor 400, isShadowPass 404,
//   tessellationEnabled 408, nearPlane 412, farPlane 416,
//   brushTextureIndex 420, brushMode 424, brushPhase 428, brushHsv 432.
// 444..448 is std140 block-size rounding.
// Flags are uint (compare with == 0u / != 0u); debugMode is int.
struct UniformObject {
    mat4 viewProjection;        // offset   0
    mat4 invViewProjection;     // offset  64
    mat4 lightSpaceMatrix;      // offset 128  cascade 0
    mat4 lightSpaceMatrix1;     // offset 192  cascade 1
    mat4 lightSpaceMatrix2;     // offset 256  cascade 2
    vec3 viewPosition;          // offset 320  camera world position
    float lightElevation;       // offset 332  lightDirection.y
    vec3 lightDirection;        // offset 336  FROM the light source toward the scene
    uint cubemapCapture;        // offset 348
    vec3 lightColor;            // offset 352
    float triplanarThreshold;   // offset 364
    float triplanarExponent;    // offset 368
    uint normalMappingEnabled;  // offset 372
    uint shadowsEnabled;        // offset 376
    int debugMode;              // offset 380
    uint roughnessEnabled;      // offset 384
    uint ambientOcclusionEnabled; // offset 388
    float tessNearDist;         // offset 392
    float tessFarDist;          // offset 396
    float tessellationFactor;   // offset 400
    uint isShadowPass;          // offset 404
    uint tessellationEnabled;   // offset 408
    float nearPlane;            // offset 412
    float farPlane;             // offset 416
    uint brushTextureIndex;     // offset 420
    uint brushMode;             // offset 424
    float brushPhase;           // offset 428
    vec3 brushHsv;              // offset 432
};

#endif // UNIFORM_OBJECT_GLSL

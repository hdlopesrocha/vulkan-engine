#pragma once
#include <cstddef>
#include <cstdint>

#include <glm/glm.hpp>

static constexpr int SHADOW_CASCADE_COUNT = 3;

// Canonical scene UBO (std140, 448 B). Single definition shared with the GLSL
// twin shaders/ubo/UniformObject.glsl — same names, fields and offsets;
// uploaded verbatim (memcpy) by MyApp/ShadowRenderer/ImpostorCapture.
// Matrices and position/light/colour vectors keep their conceptual grouping;
// independent scalars and flags are individual fields. Flags are uint32
// (GLSL bool is not a host-shareable block type).
// Member alignment is pinned with alignas to the std140 vector rules
// (vec3/vec4/mat4 -> 16) so the layout never depends on glm packing.
struct alignas(16) UniformObject {
    glm::mat4 viewProjection{1.0f};      // offset   0
    glm::mat4 invViewProjection{1.0f};   // offset  64  inverse of viewProjection (camera-constant)
    glm::mat4 lightSpaceMatrix{1.0f};    // offset 128  cascade 0
    glm::mat4 lightSpaceMatrix1{1.0f};   // offset 192  cascade 1 (4x ortho0)
    glm::mat4 lightSpaceMatrix2{1.0f};   // offset 256  cascade 2 (16x ortho0)

    alignas(16) glm::vec3 viewPosition{0.0f}; // offset 320  camera world position
    float lightElevation = 0.0f;         // offset 332  lightDirection.y (cheap shader sign test)
    alignas(16) glm::vec3 lightDirection{0.0f}; // offset 336  FROM the light source toward the scene
    uint32_t cubemapCapture = 0u;        // offset 348  1 = cubemap capture pass
    alignas(16) glm::vec3 lightColor{0.0f}; // offset 352  directional light colour
    float triplanarThreshold = 0.0f;     // offset 364
    float triplanarExponent = 0.0f;      // offset 368
    uint32_t normalMappingEnabled = 0u;  // offset 372
    uint32_t shadowsEnabled = 0u;        // offset 376  global shadow toggle
    int32_t debugMode = 0;               // offset 380  see includes/debug_modes.glsl
    uint32_t roughnessEnabled = 0u;      // offset 384
    uint32_t ambientOcclusionEnabled = 0u; // offset 388
    float tessNearDist = 0.0f;           // offset 392
    float tessFarDist = 0.0f;            // offset 396
    float tessellationFactor = 0.0f;     // offset 400
    uint32_t isShadowPass = 0u;          // offset 404  1 = shadow cascade pass
    uint32_t tessellationEnabled = 0u;   // offset 408
    float nearPlane = 0.0f;              // offset 412
    float farPlane = 0.0f;               // offset 416
    uint32_t brushTextureIndex = 0u;     // offset 420  selected brush material index
    uint32_t brushMode = 0u;             // offset 424  0 = overlay, 1 = remove, 2 = PAINT
    float brushPhase = 0.0f;             // offset 428  brush animation phase (s)
    alignas(16) glm::vec3 brushHsv{0.0f, 0.5f, 0.5f}; // offset 432
    // offset 444..448: std140 block size rounding (multiple of 16).
};
static_assert(sizeof(UniformObject) == 448, "UniformObject must be 448 bytes (std140 padding)");
static_assert(offsetof(UniformObject, viewProjection) == 0, "viewProjection offset");
static_assert(offsetof(UniformObject, invViewProjection) == 64, "invViewProjection offset");
static_assert(offsetof(UniformObject, lightSpaceMatrix) == 128, "lightSpaceMatrix offset");
static_assert(offsetof(UniformObject, lightSpaceMatrix1) == 192, "lightSpaceMatrix1 offset");
static_assert(offsetof(UniformObject, lightSpaceMatrix2) == 256, "lightSpaceMatrix2 offset");
static_assert(offsetof(UniformObject, viewPosition) == 320, "viewPosition offset");
static_assert(offsetof(UniformObject, lightElevation) == 332, "lightElevation offset");
static_assert(offsetof(UniformObject, lightDirection) == 336, "lightDirection offset");
static_assert(offsetof(UniformObject, cubemapCapture) == 348, "cubemapCapture offset");
static_assert(offsetof(UniformObject, lightColor) == 352, "lightColor offset");
static_assert(offsetof(UniformObject, triplanarThreshold) == 364, "triplanarThreshold offset");
static_assert(offsetof(UniformObject, triplanarExponent) == 368, "triplanarExponent offset");
static_assert(offsetof(UniformObject, normalMappingEnabled) == 372, "normalMappingEnabled offset");
static_assert(offsetof(UniformObject, shadowsEnabled) == 376, "shadowsEnabled offset");
static_assert(offsetof(UniformObject, debugMode) == 380, "debugMode offset");
static_assert(offsetof(UniformObject, roughnessEnabled) == 384, "roughnessEnabled offset");
static_assert(offsetof(UniformObject, ambientOcclusionEnabled) == 388, "ambientOcclusionEnabled offset");
static_assert(offsetof(UniformObject, tessNearDist) == 392, "tessNearDist offset");
static_assert(offsetof(UniformObject, tessFarDist) == 396, "tessFarDist offset");
static_assert(offsetof(UniformObject, tessellationFactor) == 400, "tessellationFactor offset");
static_assert(offsetof(UniformObject, isShadowPass) == 404, "isShadowPass offset");
static_assert(offsetof(UniformObject, tessellationEnabled) == 408, "tessellationEnabled offset");
static_assert(offsetof(UniformObject, nearPlane) == 412, "nearPlane offset");
static_assert(offsetof(UniformObject, farPlane) == 416, "farPlane offset");
static_assert(offsetof(UniformObject, brushTextureIndex) == 420, "brushTextureIndex offset");
static_assert(offsetof(UniformObject, brushMode) == 424, "brushMode offset");
static_assert(offsetof(UniformObject, brushPhase) == 428, "brushPhase offset");
static_assert(offsetof(UniformObject, brushHsv) == 432, "brushHsv offset");

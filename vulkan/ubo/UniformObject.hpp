#pragma once
#include <cstddef>
#include <cstdint>

#include <glm/glm.hpp>

// Canonical scene UBO (std140, 288 B). Single definition shared with the GLSL
// twin shaders/ubo/UniformObject.glsl — same names, fields and offsets;
// uploaded verbatim (memcpy) by MyApp/ImpostorCapture.
// Matrices and position/light/colour vectors keep their conceptual grouping;
// independent scalars and flags are individual fields. Flags are uint32
// (GLSL bool is not a host-shareable block type).
// Member alignment is pinned with alignas to the std140 vector rules
// (vec3/vec4/mat4 -> 16) so the layout never depends on glm packing.
//
// Directional shadows are evaluated by SDF raymarching in the octree path;
// there are no per-cascade light-space matrices or shadow-map parameters in
// this block anymore.
struct alignas(16) UniformObject {
    glm::mat4 viewProjection{1.0f};      // offset   0
    glm::mat4 invViewProjection{1.0f};   // offset  64  inverse of viewProjection (camera-constant)

    alignas(16) glm::vec3 viewPosition{0.0f}; // offset 128  camera world position
    float lightElevation = 0.0f;         // offset 140  lightDirection.y (cheap shader sign test)
    alignas(16) glm::vec3 lightDirection{0.0f}; // offset 144  FROM the light source toward the scene
    uint32_t cubemapCapture = 0u;        // offset 156  1 = cubemap capture pass
    alignas(16) glm::vec3 lightColor{0.0f}; // offset 160  directional light colour
    float triplanarThreshold = 0.0f;     // offset 172
    float triplanarExponent = 0.0f;      // offset 176
    uint32_t normalMappingEnabled = 0u;  // offset 180
    uint32_t shadowsEnabled = 0u;        // offset 184  global shadow toggle
    int32_t debugMode = 0;               // offset 188  see includes/DebugModes.glsl
    uint32_t roughnessEnabled = 0u;      // offset 192
    uint32_t ambientOcclusionEnabled = 0u; // offset 196
    float tessNearDist = 0.0f;           // offset 200
    float tessFarDist = 0.0f;            // offset 204
    float tessellationFactor = 0.0f;     // offset 208
    uint32_t tessellationEnabled = 0u;   // offset 212
    float nearPlane = 0.0f;              // offset 216
    float farPlane = 0.0f;               // offset 220
    uint32_t brushTextureIndex = 0u;     // offset 224  selected brush material index
    uint32_t brushMode = 0u;             // offset 228  0 = overlay, 1 = remove, 2 = PAINT
    float brushPhase = 0.0f;             // offset 232  brush animation phase (s)
    alignas(16) glm::vec3 brushHsv{0.0f, 0.5f, 0.5f}; // offset 240
    // Screen-space octree raymarch hit tolerance in pixels (0 = off).
    float raymarchScreenEps = 0.0f;      // offset 252
    // True raymarched displacement master switches. The material-level
    // strength/amplitude still gates each material; these only let the
    // renderer A/B the new path and its SDF ambient occlusion.
    uint32_t displacementEnabled = 0u;       // offset 256
    uint32_t displacementAOEnabled = 0u;     // offset 260
    // Largest |displacement| among the uploaded materials (world units). The
    // raymarch expands each layer's conservative surface AABB by this bound
    // instead of falling back to the whole root cube.
    float displacementMaxBound = 0.0f;       // offset 264
    // SDF shadow raymarch controls (octree path). The shadow ray is locked to
    // the primary hit's effective LOD; these only bound cost and penumbra.
    float shadowSoftness = 0.0f;          // offset 268  0 = hard, k in min(v, k*d/t)
    float shadowMaxDistance = 0.0f;       // offset 272  0 = far plane / scene default
    uint32_t shadowMaxSteps = 256u;       // offset 276  early-out budget (<= shader cap)
    uint32_t shadowReserved = 0u;         // offset 280
};
static_assert(sizeof(UniformObject) == 288, "UniformObject must be 288 bytes (std140)");
static_assert(offsetof(UniformObject, viewProjection) == 0, "viewProjection offset");
static_assert(offsetof(UniformObject, invViewProjection) == 64, "invViewProjection offset");
static_assert(offsetof(UniformObject, viewPosition) == 128, "viewPosition offset");
static_assert(offsetof(UniformObject, lightElevation) == 140, "lightElevation offset");
static_assert(offsetof(UniformObject, lightDirection) == 144, "lightDirection offset");
static_assert(offsetof(UniformObject, cubemapCapture) == 156, "cubemapCapture offset");
static_assert(offsetof(UniformObject, lightColor) == 160, "lightColor offset");
static_assert(offsetof(UniformObject, triplanarThreshold) == 172, "triplanarThreshold offset");
static_assert(offsetof(UniformObject, triplanarExponent) == 176, "triplanarExponent offset");
static_assert(offsetof(UniformObject, normalMappingEnabled) == 180, "normalMappingEnabled offset");
static_assert(offsetof(UniformObject, shadowsEnabled) == 184, "shadowsEnabled offset");
static_assert(offsetof(UniformObject, debugMode) == 188, "debugMode offset");
static_assert(offsetof(UniformObject, roughnessEnabled) == 192, "roughnessEnabled offset");
static_assert(offsetof(UniformObject, ambientOcclusionEnabled) == 196, "ambientOcclusionEnabled offset");
static_assert(offsetof(UniformObject, tessNearDist) == 200, "tessNearDist offset");
static_assert(offsetof(UniformObject, tessFarDist) == 204, "tessFarDist offset");
static_assert(offsetof(UniformObject, tessellationFactor) == 208, "tessellationFactor offset");
static_assert(offsetof(UniformObject, tessellationEnabled) == 212, "tessellationEnabled offset");
static_assert(offsetof(UniformObject, nearPlane) == 216, "nearPlane offset");
static_assert(offsetof(UniformObject, farPlane) == 220, "farPlane offset");
static_assert(offsetof(UniformObject, brushTextureIndex) == 224, "brushTextureIndex offset");
static_assert(offsetof(UniformObject, brushMode) == 228, "brushMode offset");
static_assert(offsetof(UniformObject, brushPhase) == 232, "brushPhase offset");
static_assert(offsetof(UniformObject, brushHsv) == 240, "brushHsv offset");
static_assert(offsetof(UniformObject, raymarchScreenEps) == 252, "raymarchScreenEps offset");
static_assert(offsetof(UniformObject, displacementEnabled) == 256, "displacementEnabled offset");
static_assert(offsetof(UniformObject, displacementAOEnabled) == 260, "displacementAOEnabled offset");
static_assert(offsetof(UniformObject, displacementMaxBound) == 264, "displacementMaxBound offset");
static_assert(offsetof(UniformObject, shadowSoftness) == 268, "shadowSoftness offset");
static_assert(offsetof(UniformObject, shadowMaxDistance) == 272, "shadowMaxDistance offset");
static_assert(offsetof(UniformObject, shadowMaxSteps) == 276, "shadowMaxSteps offset");

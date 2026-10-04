#pragma once
// Generic GPU-driven SDF framework: GPU-friendly std430 structs, all vec4-aligned.
// Independent from terrain octree (space/) and CPU meshing (sdf/*DistanceFunction).
// Only GLM + <cstdint> are included here (no Vulkan headers).
#include <glm/glm.hpp>
#include <cstdint>

// Primitive type encoded in SdfDefinitionGPU::meta.x
enum class SdfPrimitiveType : uint32_t {
    Sphere = 0,
    Box = 1,
    RoundedBox = 2,
    Capsule = 3,
    Cylinder = 4,
    Cone = 5,
    Torus = 6,
    Plane = 7,
    // Tapered, base-anchored flame: exact round cone (sphere-capped) from
    // (0,0,rBase) to (0,height,rTip) in local space, i.e. a rounded capsule
    // with two different radii. Base sits at the instance point, axis +Y
    // (aligned to the surface normal via instance euler). Spikes are added
    // procedurally in-shader (see params1).
    Flame = 8
};

// Material shading mode encoded in SdfMaterialGPU::surfaceParams.w (as float)
enum class SdfMaterialType : uint32_t {
    Surface = 0,
    Emissive = 1,
    Volume = 2,
    Transparent = 3
};

// CSG / blending operator encoded in SdfDefinitionGPU::meta.y
enum class SdfOpType : uint32_t {
    Union = 0,
    Intersection = 1,
    Subtraction = 2,
    SmoothUnion = 3,
    SmoothIntersection = 4,
    SmoothSubtraction = 5
};

// Deform flag bits encoded in SdfDefinitionGPU::meta.z
// bit0=noise, bit1=twist, bit2=bend, bit3=taper, bit4=repeat
enum class SdfDeformFlags : uint32_t {
    None = 0u,
    Noise = 1u << 0,
    Twist = 1u << 1,
    Bend = 1u << 2,
    Taper = 1u << 3,
    Repeat = 1u << 4
};

// Single SDF primitive definition.
// params0/params1 convention (local space, before instance scale):
//   Sphere:     params0.x = radius
//   Box:        params0.xyz = half extents
//   RoundedBox: params0.xyz = half extents, params1.x = corner radius
//   Capsule (Y-aligned): params0.x = radius, params0.y = half height of
//               cylindrical section (total height = 2*halfHeight + 2*radius)
//   Cylinder (Y-aligned): params0.x = radius, params0.y = half height
//   Cone (Y-aligned, centered): params0.x = base radius, params0.y = full height
//   Torus (in XZ plane): params0.x = major radius R, params0.y = minor radius r
//   Plane:      params0.xyz = normal, params0.w = offset (signed distance
//               convention: dot(n,p)+offset); bounds use a large thin box.
//   Flame:      params0.x = base radius, params0.y = flame height (base disk
//               at local y=0, tip at y=height); params1.x = tip radius
//               (0 = sharp cone tip, = base radius = capsule), params1.y =
//               spikiness amplitude (0 = smooth rounded capsule), params1.z =
//               spike frequency (tongue count), params1.w unused.
// meta.x = SdfPrimitiveType, meta.y = SdfOpType,
// meta.z = deform flags (SdfDeformFlags bits),
// meta.w = extra (bit-cast float smoothK for smooth ops).
struct SdfDefinitionGPU {
    glm::vec4 params0{0.0f};
    glm::vec4 params1{0.0f};
    glm::uvec4 meta{0u};
};
static_assert(sizeof(SdfDefinitionGPU) == 48, "SdfDefinitionGPU must be 48 bytes");

// Per-instance transform + palette indices + precomputed world AABB.
struct SdfInstanceGPU {
    glm::vec4 posScale{0.0f};   // xyz = world position, w = uniform scale
    glm::vec4 rotSeed{0.0f};    // xyz = euler angles (radians, XYZ order), w = animation seed
    glm::vec4 sizeParams{0.0f}; // x = height scale, y = radius scale, z = intensity, w = unused
    glm::uvec4 indices{0u};     // x = definition index, y = material index, z = container index, w = flags
    glm::vec4 boundsMin{0.0f};  // world AABB min, w unused
    glm::vec4 boundsMax{0.0f};  // world AABB max, w unused
};
static_assert(sizeof(SdfInstanceGPU) == 96, "SdfInstanceGPU must be 96 bytes");
static_assert(sizeof(SdfInstanceGPU) % 16 == 0, "SdfInstanceGPU must be multiple of 16");

// Shading / volumetric parameters for one material palette entry.
struct SdfMaterialGPU {
    glm::vec4 baseColor{1.0f};
    glm::vec4 surfaceParams{0.0f}; // x = roughness, y = metallic, z = opacity, w = mode (float of SdfMaterialType)
    glm::vec4 emission{0.0f};      // rgb = color, w = intensity
    glm::vec4 volumeParams{0.0f};  // x = density, y = absorption, z = scattering, w = temperature scale
    glm::vec4 extra{0.0f};         // x = smoothK, y = noiseScale, z = turbulence, w = riseSpeed
};
static_assert(sizeof(SdfMaterialGPU) == 80, "SdfMaterialGPU must be 80 bytes");
static_assert(sizeof(SdfMaterialGPU) % 16 == 0, "SdfMaterialGPU must be multiple of 16");

// Spatial container (broadphase volume) with an embedded uniform grid.
struct SdfContainerGPU {
    glm::vec4 boundsMin{0.0f};
    glm::vec4 boundsMax{0.0f};
    glm::uvec4 gridInfo{0u};   // x,y,z = resolution, w = instanceStart (offset into global index list)
    glm::uvec4 gridOffset{0u}; // x = cell buffer offset, y = index buffer offset, z = instance count, w = flags
};
static_assert(sizeof(SdfContainerGPU) == 64, "SdfContainerGPU must be 64 bytes");
static_assert(sizeof(SdfContainerGPU) % 16 == 0, "SdfContainerGPU must be multiple of 16");

// One uniform-grid cell: contiguous range [offset, offset+count) into the
// container's (or global) instance-index list.
struct SdfGridCellGPU {
    uint32_t offset = 0;
    uint32_t count = 0;
    uint32_t _pad0 = 0;
    uint32_t _pad1 = 0;
};
static_assert(sizeof(SdfGridCellGPU) == 16, "SdfGridCellGPU must be 16 bytes");

// Global raymarch / debug parameters (one small UBO).
struct SdfParamsUBO {
    glm::vec4 timeDebug{0.0f};   // x = globalTime, y = debugMode (float), z = maxSteps, w = safetyFactor
    glm::vec4 marchParams{0.0f}; // x = minStep, y = maxStep, z = hitEpsilon, w = earlyTermThreshold
    glm::vec4 fireColors0{0.0f}; // emission gradient low (kept generic)
    glm::vec4 fireColors1{0.0f}; // emission gradient high (kept generic)
};
static_assert(sizeof(SdfParamsUBO) == 64, "SdfParamsUBO must be 64 bytes");
static_assert(sizeof(SdfParamsUBO) % 16 == 0, "SdfParamsUBO must be multiple of 16");

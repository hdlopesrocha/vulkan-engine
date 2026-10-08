#pragma once
// CPU-side SDF scene builder. It stores the CANONICAL GPU-layout structs
// (sdf/types/*GPU.hpp, GLSL twins in shaders/types/*GPU.glsl) directly, so
// the renderer uploads the vectors verbatim with no conversion step.
// CPU-only helpers (flame/rock anchors/shapes, stats, config) stay separate; the
// per-container instance membership is derived from Instance::containerIdx.
// Independent from space/Octree, from sdf/*DistanceFunction, and from
// vulkan/ (no Vulkan headers here).
#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

#include "sdf/types/SdfContainer.hpp"
#include "sdf/types/SdfDefinition.hpp"
#include "sdf/types/SdfFlameAnchor.hpp"
#include "sdf/types/SdfFlameShape.hpp"
#include "sdf/types/SdfGridCell.hpp"
#include "sdf/types/SdfInstance.hpp"
#include "sdf/types/SdfMaterial.hpp"
#include "sdf/types/SdfRockAnchor.hpp"
#include "sdf/types/SdfRockShape.hpp"
#include "math/BoundingBox.hpp"

namespace sdf_gpu {

class SdfScene {
public:
    // CPU-only scene inputs stay aliased for readability.
    using FlameAnchor = sdf_gpu::FlameAnchor;
    using FlameShape = sdf_gpu::FlameShape;
    using RockAnchor = sdf_gpu::RockAnchor;
    using RockShape = sdf_gpu::RockShape;

    uint32_t addDefinition(const SdfDefinition& d);
    uint32_t addMaterial(const SdfMaterial& m);
    uint32_t addContainer(const glm::vec3& minp, const glm::vec3& maxp,
                          glm::uvec3 res = glm::uvec3(8u, 8u, 8u));
    // Stores the instance and recomputes its world AABB immediately (the
    // definition/material must already exist for an exact bound).
    uint32_t addInstance(const SdfInstance& in);
    void clear();

    // Canonical GPU-layout vectors (uploaded verbatim).
    const std::vector<SdfDefinition>& definitions() const { return definitions_; }
    std::vector<SdfDefinition>& definitions() { return definitions_; }
    const std::vector<SdfMaterial>& materials() const { return materials_; }
    std::vector<SdfMaterial>& materials() { return materials_; }
    const std::vector<SdfInstance>& instances() const { return instances_; }
    std::vector<SdfInstance>& instances() { return instances_; }
    const std::vector<SdfContainer>& containers() const { return containers_; }
    std::vector<SdfContainer>& containers() { return containers_; }
    const std::vector<SdfGridCell>& cells() const { return cells_; }
    const std::vector<uint32_t>& indices() const { return indices_; }

    // Refreshes every instance AABB and rebuilds all container grids/cells
    // into the canonical vectors (also refreshes each container's cellStart).
    // Call once after mutating the scene (before upload); builders already
    // leave the scene consistent, so this is a cheap no-op for them.
    void rebuild();

    // Conservative world-space AABB for one instance (boundsMin/boundsMax are
    // recomputed by rebuild(); addInstance fills them once up front).
    BoundingBox computeInstanceBounds(const SdfInstance& in) const;

    // Merge two scenes into one (definitions/materials/containers/instances
    // concatenated with index remap). Lets independent emitters (lava fire,
    // smoke bombs, future effects) share one GPU upload and one march.
    static SdfScene merge(const SdfScene& a, const SdfScene& b);

    // Smoke-bomb scene: 1 Smoke-sphere definition (maximum radius), 1 gray
    // volumetric material, 1 static container, 1 instance. Growth, noise,
    // bullets and render tuning live in the smoke state buffer
    // (SmokeFragBullet), so this topology never rebuilds for widget tweaks.
    static SdfScene createSmokeBomb(const glm::vec3& center, float scale,
                                     float seed = 0.0f);

    // Default fire demo: 1 flame definition, 1 volumetric fire material,
    // 1 container AABB, N instances with random pos/scale/seed
    // (std::mt19937 seeded with 1234).
    static SdfScene createFireDemo(uint32_t flameCount = 64,
                                    glm::vec3 center = glm::vec3(0.0f),
                                    float areaSize = 20.0f);

    // Fire scene from explicit anchors (tapered base-anchored flames,
    // container auto-fit, adaptive grid targeting ~6 m cells). Empty
    // anchors -> scene with no containers (renders nothing).
    static SdfScene createFireFromAnchors(const std::vector<FlameAnchor>& anchors,
                                          const FlameShape& shape);

    // Rock scene from explicit anchors (Perlin-displaced spheres, container
    // auto-fit targeting ~32 m cells). Empty anchors -> definition+material
    // only, no containers (renders nothing).
    static SdfScene createRocksFromAnchors(const std::vector<RockAnchor>& anchors,
                                           const RockShape& shape);

    // Orientation for Y-up SDF instances (flames grow along local +Y):
    // shortest-arc rotation taking +Y to the surface normal `n`, expressed
    // as XYZ euler radians in the R = Rx * Ry * Rz convention used by
    // rotationFromEuler/sdfEulerMat. Lets flames stand perpendicular to the
    // lava surface that spawned them (overhangs included), like grass
    // billboards tilting onto the smooth normal.
    static glm::vec3 eulerAlignYToNormal(const glm::vec3& n);

private:
    std::vector<SdfDefinition> definitions_;
    std::vector<SdfMaterial> materials_;
    std::vector<SdfInstance> instances_;
    std::vector<SdfContainer> containers_;
    std::vector<SdfGridCell> cells_;
    std::vector<uint32_t> indices_;
};

} // namespace sdf_gpu

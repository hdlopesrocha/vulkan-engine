#pragma once
// Generic CPU-side SDF scene for the GPU-driven SDF framework.
// Independent from space/Octree and from sdf/*DistanceFunction (not included).
// Scene description structs live in types/ (one file per struct, shared, no
// Vulkan dependency); GPU-wire flattening uses the types/Sdf*GPU structs.
#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

#include "types/SdfDefinition.hpp"
#include "types/SdfMaterial.hpp"
#include "types/SdfInstance.hpp"
#include "types/SdfContainer.hpp"
#include "types/SdfFlameAnchor.hpp"
#include "types/SdfFlameShape.hpp"
#include "vulkan/types/SdfDefinitionGPU.hpp"
#include "vulkan/types/SdfInstanceGPU.hpp"
#include "vulkan/types/SdfMaterialGPU.hpp"
#include "vulkan/types/SdfContainerGPU.hpp"
#include "vulkan/types/SdfGridCellGPU.hpp"
#include "math/BoundingBox.hpp"

namespace sdf_gpu {

class SdfScene {
public:
    // Scene description types live in types/ (one file per struct, shared
    // with renderer/widgets); aliased here so existing SdfScene::X code is
    // unaffected. Flatten machinery (BuiltGrid/FlattenedScene) stays nested:
    // it references the GPU-wire structs and belongs to the builder.
    using Definition = sdf_gpu::Definition;
    using Material = sdf_gpu::Material;
    using Instance = sdf_gpu::Instance;
    using Container = sdf_gpu::Container;
    using FlameAnchor = sdf_gpu::FlameAnchor;
    using FlameShape = sdf_gpu::FlameShape;

    struct BuiltGrid {
        std::vector<SdfGridCellGPU> cells;
        std::vector<uint32_t> indices;
    };

    struct FlattenedScene {
        std::vector<SdfDefinitionGPU> definitions;
        std::vector<SdfInstanceGPU> instances;
        std::vector<SdfMaterialGPU> materials;
        std::vector<SdfContainerGPU> containers;
        std::vector<SdfGridCellGPU> cells;
        std::vector<uint32_t> indices;
    };

    uint32_t addDefinition(const Definition& d);
    uint32_t addMaterial(const Material& m);
    uint32_t addContainer(const glm::vec3& minp, const glm::vec3& maxp,
                          glm::uvec3 res = glm::uvec3(8u, 8u, 8u));
    uint32_t addInstance(const Instance& in);
    void clear();

    // Accessors (const and mutable).
    const std::vector<Definition>& definitions() const { return definitions_; }
    std::vector<Definition>& definitions() { return definitions_; }
    const std::vector<Material>& materials() const { return materials_; }
    std::vector<Material>& materials() { return materials_; }
    const std::vector<Instance>& instances() const { return instances_; }
    std::vector<Instance>& instances() { return instances_; }
    const std::vector<Container>& containers() const { return containers_; }
    std::vector<Container>& containers() { return containers_; }

    // Conservative world-space AABB per instance (w == 0).
    // Extents derive from the definition type scaled by instance scale,
    // height/radius scales, rotated by euler, then padded generously for
    // deformation: pad = 0.5 + turbulence*0.5 (+ smoothK for smooth ops,
    // plus small extras for twist/bend/taper/repeat bits when set).
    BoundingBox computeInstanceBounds(const Instance& in) const;

    // Uniform grid for one container: cells = res.x*res.y*res.z with
    // offset/count into a contiguous index buffer. Each instance is inserted
    // into every cell its AABB overlaps (clamped to the container).
    // Cell index = x + res.x * (y + res.y * z).
    BuiltGrid buildContainerGrid(uint32_t containerIdx) const;

    // Flatten helpers: fill caller-provided GPU vectors.
    void flattenDefinitions(std::vector<SdfDefinitionGPU>& out) const;
    void flattenMaterials(std::vector<SdfMaterialGPU>& out) const;
    void flattenInstances(std::vector<SdfInstanceGPU>& out) const;
    void flattenContainers(std::vector<SdfContainerGPU>& out) const;
    // Concatenated per-container grids (cell offsets are per-container local
    // ranges rebased into the global arrays; see flatten() for global offsets).
    void flattenGrid(std::vector<SdfGridCellGPU>& outCells,
                     std::vector<uint32_t>& outIndices) const;
    // Full flatten with consistent container gridInfo/gridOffset values:
    // gridInfo.w = index start, gridOffset = (cellStart, indexStart, count, 0).
    FlattenedScene flatten() const;

    // Merge two scenes into one (definitions/materials/containers/instances
    // concatenated with index remap). Lets independent emitters (lava fire,
    // smoke bombs, future effects) share one GPU upload and one march.
    static SdfScene merge(const SdfScene& a, const SdfScene& b);

    // Smoke-bomb scene: 1 Smoke-sphere definition (maximum radius), 1 gray
    // volumetric material, 1 static container, 1 instance. Growth, noise,
    // bullets and render tuning live in the smoke state buffer, so this
    // topology never needs rebuilding for widget tweaks.
    static SdfScene createSmokeBomb(const glm::vec3& center, float maxRadius,
                                     float seed = 0.0f);

    // Default fire demo: 1 capsule flame definition, 1 volumetric fire
    // material, 1 container AABB, N instances with random pos/scale/seed
    // (std::mt19937 seeded with 1234).
    static SdfScene createFireDemo(uint32_t flameCount = 64,
                                    glm::vec3 center = glm::vec3(0.0f),
                                    float areaSize = 20.0f);

    // Fire scene from explicit anchors (tapered base-anchored flames,
    // container auto-fit, adaptive grid targeting ~6 m cells). Empty
    // anchors -> scene with no containers (renders nothing).
    static SdfScene createFireFromAnchors(const std::vector<FlameAnchor>& anchors,
                                          const FlameShape& shape);

    // Orientation for Y-up SDF instances (flames grow along local +Y):
    // shortest-arc rotation taking +Y to the surface normal `n`, expressed
    // as XYZ euler radians in the R = Rx * Ry * Rz convention used by
    // rotationFromEuler/sdfEulerMat. Lets flames stand perpendicular to the
    // lava surface that spawned them (overhangs included), like grass
    // billboards tilting onto the smooth normal.
    static glm::vec3 eulerAlignYToNormal(const glm::vec3& n);

private:
    std::vector<Definition> definitions_;
    std::vector<Material> materials_;
    std::vector<Instance> instances_;
    std::vector<Container> containers_;
};

} // namespace sdf_gpu

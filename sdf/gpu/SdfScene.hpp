#pragma once
// Generic CPU-side SDF scene for the GPU-driven SDF framework.
// Independent from space/Octree and from sdf/*DistanceFunction (not included).
// Only standard C++23 + GLM + vulkan/ubo/SdfUBO.hpp are used.
#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

#include "vulkan/ubo/SdfUBO.hpp"

namespace sdf_gpu {

class SdfScene {
public:
    struct Definition {
        SdfPrimitiveType prim = SdfPrimitiveType::Sphere;
        SdfOpType op = SdfOpType::Union;
        glm::vec4 params0 = glm::vec4(0.0f);
        glm::vec4 params1 = glm::vec4(0.0f);
        uint32_t deformFlags = 0;
        float smoothK = 0.5f;
    };

    struct Material {
        SdfMaterialType mode = SdfMaterialType::Volume;
        glm::vec4 baseColor = glm::vec4(1.0f);
        float roughness = 0.5f;
        float metallic = 0.0f;
        float opacity = 1.0f;
        glm::vec4 emission = glm::vec4(1.0f, 0.5f, 0.1f, 2.0f);
        float density = 1.0f;
        float absorption = 0.5f;
        float scattering = 0.5f;
        float tempScale = 1.0f;
        float noiseScale = 2.5f;
        float turbulence = 0.6f;
        float riseSpeed = 1.5f;
    };

    struct Instance {
        uint32_t defIdx = 0;
        uint32_t matIdx = 0;
        glm::vec3 pos = glm::vec3(0.0f);
        glm::vec3 euler = glm::vec3(0.0f);
        float scale = 1.0f;
        float heightScale = 1.0f;
        float radiusScale = 1.0f;
        float intensity = 1.0f;
        float seed = 0.0f;
        uint32_t containerIdx = 0;
    };

    struct Container {
        glm::vec3 minp = glm::vec3(0.0f);
        glm::vec3 maxp = glm::vec3(1.0f);
        glm::uvec3 resolution = glm::uvec3(8u, 8u, 8u);
        std::vector<uint32_t> instanceIndices;
    };

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
    glm::vec4 computeInstanceBoundsMin(const Instance& in) const;
    glm::vec4 computeInstanceBoundsMax(const Instance& in) const;

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

    // Default fire demo: 1 capsule flame definition, 1 volumetric fire
    // material, 1 container AABB, N instances with random pos/scale/seed
    // (std::mt19937 seeded with 1234).
    static SdfScene createFireDemo(uint32_t flameCount = 64,
                                    glm::vec3 center = glm::vec3(0.0f),
                                    float areaSize = 20.0f);

    // One flame anchor: world position + per-instance variation. Produced by
    // the lava collector (brush-4 triangles) or any other emitter; the SDF
    // scene itself stays independent of terrain/octree representations.
    struct FlameAnchor {
        glm::vec3 pos = glm::vec3(0.0f);
        glm::vec3 euler = glm::vec3(0.0f); // R = Rx * Ry * Rz, local +Y = flame axis
        float scale = 1.0f;
        float heightScale = 1.0f; // per-instance vertical stretch
        float seed = 0.0f;
        float intensity = 1.0f;
    };
    // Fire scene from explicit anchors: tapered base-anchored flame
    // definition, volumetric fire material, container auto-fit to the
    // anchors (padded for flame height + deformation) with an adaptive
    // uniform grid targeting ~6 m cells (clamped to 1..24 per axis). Empty
    // anchors -> scene with no containers (renders nothing).
    struct FlameShape {
        float baseRadius = 1.0f;  // local units, scaled by instance scale
        float height = 3.2f;     // base disk (y=0) to tip, local units
        float tipRadius = 0.25f; // 0 = sharp cone tip, = base = capsule
        float spikiness = 0.35f; // spike amplitude, 0 = smooth rounded capsule
        float spikeFreq = 2.0f;  // tongue count around the axis
        float density = 0.35f;   // volumetric density multiplier (lower = glassier)
    };
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

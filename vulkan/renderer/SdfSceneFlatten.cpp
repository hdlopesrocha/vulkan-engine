// Vulkan-side SDF scene flattening (moved verbatim out of SdfScene so the
// scene class stays free of vulkan/; only public SdfScene API is used).
#include "vulkan/renderer/SdfSceneFlatten.hpp"

#include <bit>
#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

namespace {
using sdf_gpu::SdfScene;

void flattenDefinitions(const SdfScene& scene, std::vector<SdfDefinitionGPU>& out) {
    const auto& definitions = scene.definitions();
    out.resize(definitions.size());
    for (size_t i = 0; i < definitions.size(); ++i) {
        const SdfScene::Definition& d = definitions[i];
        out[i].params0 = d.params0;
        out[i].params1 = d.params1;
        out[i].meta = glm::uvec4(static_cast<uint32_t>(d.prim), static_cast<uint32_t>(d.op), d.deformFlags,
                                 std::bit_cast<uint32_t>(d.smoothK));
    }
}

void flattenMaterials(const SdfScene& scene, std::vector<SdfMaterialGPU>& out) {
    const auto& materials = scene.materials();
    out.resize(materials.size());
    for (size_t i = 0; i < materials.size(); ++i) {
        const SdfScene::Material& m = materials[i];
        out[i].baseColor = m.baseColor;
        out[i].surfaceParams = glm::vec4(m.roughness, m.metallic, m.opacity, static_cast<float>(static_cast<uint32_t>(m.mode)));
        out[i].emission = m.emission;
        out[i].volumeParams = glm::vec4(m.density, m.absorption, m.scattering, m.tempScale);
        // No per-material smoothK field exists on the CPU Material; store the
        // framework default (matches Definition::smoothK default).
        out[i].extra = glm::vec4(0.5f, m.noiseScale, m.turbulence, m.riseSpeed);
    }
}

void flattenInstances(const SdfScene& scene, std::vector<SdfInstanceGPU>& out) {
    const auto& instances = scene.instances();
    out.resize(instances.size());
    for (size_t i = 0; i < instances.size(); ++i) {
        const SdfScene::Instance& in = instances[i];
        out[i].posScale = glm::vec4(in.pos, in.scale);
        out[i].rotSeed = glm::vec4(in.euler, in.seed);
        out[i].sizeParams = glm::vec4(in.heightScale, in.radiusScale, in.intensity, 0.0f);
        out[i].indices = glm::uvec4(in.defIdx, in.matIdx, in.containerIdx, 0u);
        BoundingBox b = scene.computeInstanceBounds(in);
        out[i].boundsMin = glm::vec4(b.getMin(), 0.0f);
        out[i].boundsMax = glm::vec4(b.getMax(), 0.0f);
    }
}

} // namespace

SdfFlattenedScene FlattenSdfScene(const sdf_gpu::SdfScene& scene) {
    SdfFlattenedScene f;
    flattenDefinitions(scene, f.definitions);
    flattenMaterials(scene, f.materials);
    flattenInstances(scene, f.instances);
    const auto& containers = scene.containers();
    const auto& instances = scene.instances();
    f.containers.resize(containers.size());
    f.cells.clear();
    f.indices.clear();
    for (uint32_t ci = 0; ci < static_cast<uint32_t>(containers.size()); ++ci) {
        const SdfScene::Container& c = containers[ci];
        sdf_gpu::SdfUniformGrid g = scene.buildContainerGrid(ci);
        const uint32_t cellStart = static_cast<uint32_t>(f.cells.size());
        const uint32_t indexStart = static_cast<uint32_t>(f.indices.size());
        // Count members for gridOffset.z (same union rule as the grid).
        uint32_t count = 0;
        {
            std::vector<char> seen(instances.size(), 0);
            for (uint32_t gi : c.instanceIndices) {
                if (gi < instances.size() && !seen[gi]) {
                    seen[gi] = 1;
                    ++count;
                }
            }
            for (uint32_t gi = 0; gi < static_cast<uint32_t>(instances.size()); ++gi) {
                if (instances[gi].containerIdx == ci && !seen[gi]) {
                    seen[gi] = 1;
                    ++count;
                }
            }
        }
        f.containers[ci].boundsMin = glm::vec4(c.aabb.getMinX(), c.aabb.getMinY(), c.aabb.getMinZ(), 0.0f);
        f.containers[ci].boundsMax = glm::vec4(c.aabb.getMaxX(), c.aabb.getMaxY(), c.aabb.getMaxZ(), 0.0f);
        f.containers[ci].gridInfo = glm::uvec4(c.resolution.x, c.resolution.y, c.resolution.z, indexStart);
        f.containers[ci].gridOffset = glm::uvec4(cellStart, indexStart, count, 0u);
        // Widen plain grid cells to the GPU wire format, rebasing per-container
        // local offsets into the global index array.
        for (const auto& cell : g.cells) {
            SdfGridCellGPU gpuCell;
            gpuCell.offset = cell.offset + indexStart;
            gpuCell.count = cell.count;
            f.cells.push_back(gpuCell);
        }
        f.indices.insert(f.indices.end(), g.indices.begin(), g.indices.end());
    }
    return f;
}

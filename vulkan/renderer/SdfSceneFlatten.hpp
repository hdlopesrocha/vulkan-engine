#pragma once

// Vulkan-side adapter: flattens a CPU sdf_gpu::SdfScene into the GPU-wire
// vectors (vulkan/types/*GPU) consumed by SdfRenderer's SSBO mirrors.
// Lives under vulkan/ so the headless server build (which excludes
// vulkan/**.o) never links it; the scene class itself (sdf/types/SdfScene)
// has no Vulkan dependency.
#include "sdf/types/SdfScene.hpp"
#include "vulkan/types/SdfContainerGPU.hpp"
#include "vulkan/types/SdfDefinitionGPU.hpp"
#include "vulkan/types/SdfGridCellGPU.hpp"
#include "vulkan/types/SdfInstanceGPU.hpp"
#include "vulkan/types/SdfMaterialGPU.hpp"

#include <cstdint>
#include <vector>

struct SdfFlattenedScene {
    std::vector<SdfDefinitionGPU> definitions;
    std::vector<SdfInstanceGPU> instances;
    std::vector<SdfMaterialGPU> materials;
    std::vector<SdfContainerGPU> containers;
    std::vector<SdfGridCellGPU> cells;
    std::vector<uint32_t> indices;
};

// Full flatten with consistent container gridInfo/gridOffset values:
// gridInfo.w = index start, gridOffset = (cellStart, indexStart, count, 0).
// Per-container grid cells are local ranges rebased into the global arrays.
SdfFlattenedScene FlattenSdfScene(const sdf_gpu::SdfScene& scene);

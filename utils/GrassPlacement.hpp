#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>

#include "../math/geometry/Geometry.hpp"

// Deterministic grass placement shared by VegetationRenderer's CPU fallback
// path and the offline exporters. Vulkan-free on purpose: the Blender export
// runs headless and must reproduce the renderer's tufts without a device.
//
// The per-instance math (hashes, PRNG, biome gate, dominant-vertex brush test)
// is the CPU twin of shaders/computation/vegetation/VegetationPlacement.comp;
// keep them in sync. The runtime GPU path expands the chunk mesh into
// area-weighted slots with a deterministic per-triangle hash instead of this
// file's mt19937/shuffle, so the exact slot set can differ; the tuft placement
// rule itself is identical.

namespace grass {

// LandBrush::grass material index (utils/LandBrush.cpp).
constexpr int kGrassBrush = 3;
// Instances per world-space unit of triangle area.
constexpr float kDensity = 0.01f;

struct Instance {
    glm::vec3 position{0.0f};
    glm::vec3 normal{0.0f, 1.0f, 0.0f};
    int variant = 0;
    float rotation = 0.0f; // 0..1 fraction of a full turn around +Y
};

uint32_t xorshift32(uint32_t& state);
float randFloat(uint32_t& state);
uint32_t posHash(const glm::vec3& p);
uint32_t cellHash(glm::ivec2 c);
float biomeNoise(const glm::vec2& xz);

// 2D Perlin, exact CPU twin of shaders/includes/noise/Perlin2D.glsl.
float perlin2(const glm::vec2& p);

// Per-instance height scale in [0.6, 1.4], CPU twin of
// vegetationHeightScale() (shaders/includes/vegetation/VegetationCommon.glsl).
float heightScaleAt(const glm::vec2& worldXZ);

// Expands the chunk's triangles into area-weighted virtual instance slots
// (unbiased stochastic rounding per triangle, chunk-seeded shuffle). `out`
// holds 3 indices per slot.
void collectSlots(const Geometry& geom, uintptr_t nodeId, std::vector<uint32_t>& out);

// Resolves slots into tufts (barycentric sample, dominant-vertex brush test,
// biome gate, variant/rotation). `seed` is the low 32 bits of the chunk id,
// as used by the renderer.
void placeInstances(const std::vector<glm::vec3>& positions,
                    const std::vector<glm::vec3>& normals,
                    const std::vector<int>& brushes,
                    const std::vector<uint32_t>& slots,
                    uint32_t billboardCount,
                    uint32_t seed,
                    std::vector<Instance>& out);

// Convenience overload for offline consumers: builds the parallel vertex
// arrays and the low-32 seed from the node pointer exactly like
// VegetationRenderer::generateChunkInstancesCPU.
void placeInstances(const Geometry& geom, const std::vector<uint32_t>& slots,
                    uintptr_t nodeId, uint32_t billboardCount,
                    std::vector<Instance>& out);

} // namespace grass

#include "LandBrush.hpp"

#include <algorithm>
#include <cmath>

#include "GrassPlacement.hpp" // perlin2 (exact CPU twin of Perlin2D.glsl)
#include "../math/BrushMode.hpp"

namespace landbrush {
namespace {

// Palette / mixer indices (mirror SceneMaterialTable.cpp and LandBrush.glsl).
constexpr int kSourceCount = 26;
constexpr int kMixerFirst = kSourceCount;

constexpr int kGrassMixSand = kMixerFirst + 0;
constexpr int kGrassMixSnow = kMixerFirst + 1;
constexpr int kRockMixGrass = kMixerFirst + 2;
constexpr int kRockMixSnow = kMixerFirst + 3;
constexpr int kRockMixSand = kMixerFirst + 4;

constexpr int kBiomeGrass = 3;
constexpr int kBiomeGroundGrass = 22;
constexpr int kBiomeLeaves = 2;
constexpr int kBiomeDirt = 1;
constexpr int kBiomeBareDirt = 21;
constexpr int kBiomeMixFirst = kMixerFirst + 5;

constexpr int kRockBiome006 = 17;
constexpr int kRockBiome018 = 24;
constexpr int kRockBiome001 = 25;
constexpr int kRockBiomeMixFirst = kBiomeMixFirst + 4;

float landBiomeShape(const glm::vec2& p, const glm::vec2& warpP) {
    const float base = grass::perlin2(p)
        + 0.5f * grass::perlin2(p * 2.07f + glm::vec2(19.3f, -7.1f));
    const float warp = grass::perlin2(warpP);
    const float n = std::clamp((base * 0.6667f + warp * 0.12f) * 1.9f,
                               -1.0f, 1.0f);
    return 0.5f + 0.5f * std::tanh(2.5f * n);
}

float landBiomeField(const glm::vec2& xz) {
    return landBiomeShape(xz * 0.01f, xz * 0.075f + glm::vec2(4.7f, 23.9f));
}

float landRockField(const glm::vec2& xz) {
    return landBiomeShape(xz * 0.006f + glm::vec2(11.7f, -3.9f),
                          xz * 0.055f + glm::vec2(31.2f, -17.6f));
}

int landBiomeBase(int band) {
    if (band <= 0) return kBiomeGrass;
    if (band == 1) return kBiomeGroundGrass;
    if (band == 2) return kBiomeLeaves;
    if (band == 3) return kBiomeDirt;
    return kBiomeBareDirt;
}

int landBiomeMaterial(const glm::vec2& xz) {
    constexpr int kBiomeCount = 5;
    constexpr float kBlend = 0.15f;

    const float t = landBiomeField(xz) * static_cast<float>(kBiomeCount);
    const int band = std::clamp(static_cast<int>(std::floor(t)), 0, kBiomeCount - 1);
    const float f = t - static_cast<float>(band);

    if (band + 1 < kBiomeCount && f > 1.0f - kBlend) return kBiomeMixFirst + band;
    if (band > 0 && f < kBlend) return kBiomeMixFirst + (band - 1);
    return landBiomeBase(band);
}

int landRockBase(int band) {
    if (band <= 0) return kRock;
    if (band == 1) return kRockBiome006;
    if (band == 2) return kRockBiome018;
    return kRockBiome001;
}

int landRockMaterial(const glm::vec2& xz) {
    constexpr int kRockCount = 4;
    constexpr float kBlend = 0.18f;

    const float t = landRockField(xz) * static_cast<float>(kRockCount);
    const int band = std::clamp(static_cast<int>(std::floor(t)), 0, kRockCount - 1);
    const float f = t - static_cast<float>(band);

    if (band + 1 < kRockCount && f > 1.0f - kBlend) return kRockBiomeMixFirst + band;
    if (band > 0 && f < kBlend) return kRockBiomeMixFirst + (band - 1);
    return landRockBase(band);
}

} // namespace

int materialIndex(const glm::vec3& pos, const glm::vec3& normal) {
    const glm::vec3 n = glm::normalize(normal);
    const glm::vec2 xz(pos.x, pos.z);
    const float steepness = n.y;
    constexpr float kGrassLevel = 256.0f;
    constexpr float kSandLevel = 16.0f;
    constexpr float kSoftSandLevel = 2.0f;

    if (steepness <= 0.01f) return kUnderground;
    if (steepness < 0.980f) return landRockMaterial(xz);
    if (steepness < 0.985f) {
        if (pos.y < kSoftSandLevel) return landRockMaterial(xz);
        if (pos.y < kSandLevel) return kRockMixSand;
        if (pos.y < kGrassLevel) return kRockMixGrass;
        return kRockMixSnow;
    }
    if (pos.y < kSoftSandLevel) return kSoftSand;
    if (pos.y < kSandLevel) return kSand;
    if (pos.y < kSandLevel + 4.0f) return kGrassMixSand;
    if (pos.y < kGrassLevel) return landBiomeMaterial(xz);
    if (pos.y < kGrassLevel + 32.0f) return kGrassMixSnow;
    return kSnow;
}

int resolveVertexMaterial(int brush, const glm::vec3& pos,
                          const glm::vec3& normal) {
    if (brush == kLandBrushIndex) {
        const int classified = materialIndex(pos, normal);
        // The GPU writes LAND_ROCK for underground samples so triangles that
        // straddle the contour still carry a real material.
        return (classified == kUnderground) ? kRock : classified;
    }
    if (brush == kNormalBrushIndex) {
        const glm::vec3 n = glm::normalize(normal);
        return (glm::dot(n, glm::vec3(0.0f, 1.0f, 0.0f)) > 0.0f)
            ? 0 : DISCARD_BRUSH_INDEX;
    }
    return brush;
}

} // namespace landbrush

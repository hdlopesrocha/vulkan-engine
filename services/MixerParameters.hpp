#pragma once

#include <cstddef>
#include <cstdint>

// Per-mixer parameters for the Perlin generation compute pass.
struct MixerParameters {
    size_t targetLayer;
    uint primaryTextureIdx;
    uint secondaryTextureIdx;
    // Default Perlin parameters (kept here for compatibility)
    float perlinScale = 8.0f;
    float perlinOctaves = 4.0f;
    float perlinPersistence = 0.5f;
    float perlinLacunarity = 2.0f;
    float perlinBrightness = 0.0f;  // -1.0 to 1.0
    float perlinContrast = 5.0f;    // 0.0 to 5.0 (lower default gives smoother blends)
    uint32_t perlinSeed = 12345;    // Fixed seed for consistent generation
    float perlinTime = 0.0f;        // Time parameter for noise evolution
};

#pragma once

#include <cstdint>

// One generated layer: a Perlin-noise blend of the primary and secondary
// source layers (shaders/computation/TextureMixer.comp). Target index is the
// layer it fills; sources are source-layer indices.
struct SceneMixerDesc {
    uint32_t target = 0;
    uint32_t primary = 0;
    uint32_t secondary = 0;
    float perlinScale = 8.0f;
    float perlinOctaves = 4.0f;
    float perlinPersistence = 0.5f;
    float perlinLacunarity = 2.0f;
    float perlinBrightness = 0.0f;
    float perlinContrast = 5.0f;
    uint32_t perlinSeed = 12345;
    float perlinTime = 0.0f;
};
